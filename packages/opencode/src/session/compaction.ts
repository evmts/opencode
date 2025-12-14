import { BusEvent } from "@/bus/bus-event"
import { Bus } from "@/bus"
import { generateObject, type ModelMessage } from "ai"
import { Session } from "."
import { Identifier } from "../id/id"
import { Instance } from "../project/instance"
import { Provider } from "../provider/provider"
import { MessageV2 } from "./message-v2"
import { SystemPrompt } from "./system"
import z from "zod"
import { SessionPrompt } from "./prompt"
import { Flag } from "../flag/flag"
import { Token } from "../util/token"
import { Config } from "../config/config"
import { Log } from "../util/log"
import { ProviderTransform } from "@/provider/transform"
import { fn } from "@/util/fn"
import { mergeDeep, pipe } from "remeda"
import path from "path"
import fs from "fs/promises"

export namespace SessionCompaction {
  const log = Log.create({ service: "session.compaction" })

  export const Event = {
    Compacted: BusEvent.define(
      "session.compacted",
      z.object({
        sessionID: z.string(),
      }),
    ),
  }

  export function isOverflow(input: { tokens: MessageV2.Assistant["tokens"]; model: Provider.Model }) {
    if (Flag.OPENCODE_DISABLE_AUTOCOMPACT) return false
    const context = input.model.limit.context
    if (context === 0) return false
    const count = input.tokens.input + input.tokens.cache.read + input.tokens.output
    const output = Math.min(input.model.limit.output, SessionPrompt.OUTPUT_TOKEN_MAX) || SessionPrompt.OUTPUT_TOKEN_MAX
    const usable = context - output
    return count > usable
  }

  export const PRUNE_MINIMUM = 20_000
  export const PRUNE_PROTECT = 40_000

  // goes backwards through parts until there are 40_000 tokens worth of tool
  // calls. then erases output of previous tool calls. idea is to throw away old
  // tool calls that are no longer relevant.
  export async function prune(input: { sessionID: string }) {
    if (Flag.OPENCODE_DISABLE_PRUNE) return
    log.info("pruning")
    const msgs = await Session.messages({ sessionID: input.sessionID })
    let total = 0
    let pruned = 0
    const toPrune = []
    let turns = 0

    loop: for (let msgIndex = msgs.length - 1; msgIndex >= 0; msgIndex--) {
      const msg = msgs[msgIndex]
      if (msg.info.role === "user") turns++
      if (turns < 2) continue
      if (msg.info.role === "assistant" && msg.info.summary) break loop
      for (let partIndex = msg.parts.length - 1; partIndex >= 0; partIndex--) {
        const part = msg.parts[partIndex]
        if (part.type === "tool")
          if (part.state.status === "completed") {
            if (part.state.time.compacted) break loop
            const estimate = Token.estimate(part.state.output)
            total += estimate
            if (total > PRUNE_PROTECT) {
              pruned += estimate
              toPrune.push(part)
            }
          }
      }
    }
    log.info("found", { pruned, total })
    if (pruned > PRUNE_MINIMUM) {
      for (const part of toPrune) {
        if (part.state.status === "completed") {
          part.state.time.compacted = Date.now()
          await Session.updatePart(part)
        }
      }
      log.info("pruned", { count: toPrune.length })
    }
  }

  const CompactionSchema = z.object({
    summary: z
      .string()
      .describe(
        "Comprehensive handoff: files changed, decisions, errors, user preferences, implementation details. Include everything needed to continue.",
      ),
    continue: z
      .string()
      .describe(
        "Focused instruction for immediate next steps. Include task and current direction (what's been tried/ruled out, what approach to take). This becomes the user message that resumes work.",
      ),
    files: z
      .array(
        z.object({
          path: z.string().describe("Relative path within .agent-files/ (e.g. 'STATUS.md', 'notes/debug.md')"),
          content: z.string().describe("Full file content"),
        }),
      )
      .optional()
      .default([])
      .describe(
        "Files to write to .agent-files/ directory. Always include STATUS.md with current state. Add other files as needed for context that should persist across sessions.",
      ),
  })

  export async function process(input: {
    parentID: string
    messages: MessageV2.WithParts[]
    sessionID: string
    model: {
      providerID: string
      modelID: string
    }
    agent: string
    abort: AbortSignal
    trigger: MessageV2.CompactionTrigger
  }) {
    const cfg = await Config.get()

    // Use configured compaction model if specified, otherwise use the input model
    const compactionConfig = cfg.compaction
    let modelInfo = input.model
    if (compactionConfig?.model) {
      const [providerID, modelID] = compactionConfig.model.split("/")
      if (providerID && modelID) {
        modelInfo = { providerID, modelID }
      }
    }

    const model = await Provider.getModel(modelInfo.providerID, modelInfo.modelID)
    const language = await Provider.getLanguage(model)
    const system = SystemPrompt.compaction(model.providerID, compactionConfig?.system_prompt)
    const userPrompt =
      compactionConfig?.user_prompt ?? "Summarize the conversation and create a handoff for continuing work."

    const msg = (await Session.updateMessage({
      id: Identifier.ascending("message"),
      role: "assistant",
      parentID: input.parentID,
      sessionID: input.sessionID,
      mode: input.agent,
      summary: true,
      path: {
        cwd: Instance.directory,
        root: Instance.worktree,
      },
      cost: 0,
      tokens: {
        output: 0,
        input: 0,
        reasoning: 0,
        cache: { read: 0, write: 0 },
      },
      modelID: modelInfo.modelID,
      providerID: model.providerID,
      time: {
        created: Date.now(),
      },
    })) as MessageV2.Assistant

    const options = pipe({}, mergeDeep(ProviderTransform.options(model, input.sessionID)), mergeDeep(model.options))

    try {
      const result = await generateObject({
        schema: CompactionSchema,
        providerOptions: ProviderTransform.providerOptions(model, options),
        headers: model.headers,
        abortSignal: input.abort,
        messages: [
          ...system.map(
            (x): ModelMessage => ({
              role: "system",
              content: x,
            }),
          ),
          ...MessageV2.toModelMessage(
            input.messages.filter((m) => {
              if (m.info.role !== "assistant" || m.info.error === undefined) {
                return true
              }
              if (
                MessageV2.AbortedError.isInstance(m.info.error) &&
                m.parts.some((part) => part.type !== "step-start" && part.type !== "reasoning")
              ) {
                return true
              }
              return false
            }),
          ),
          {
            role: "user",
            content: userPrompt,
          },
        ],
        model: language,
        experimental_telemetry: {
          isEnabled: cfg.experimental?.openTelemetry,
          metadata: {
            userId: cfg.username ?? "unknown",
            sessionId: input.sessionID,
          },
        },
      })

      const usage = Session.getUsage({
        model,
        usage: result.usage,
        metadata: result.providerMetadata,
      })
      msg.cost = usage.cost
      msg.tokens = usage.tokens
      msg.finish = "stop"
      msg.time.completed = Date.now()
      await Session.updateMessage(msg)

      // Store handoff prompt in session
      await Session.update(input.sessionID, (draft) => {
        draft.handoff = {
          prompt: result.object.continue,
          createdAt: Date.now(),
          trigger: input.trigger,
        }
      })

      // Write agent files
      if (result.object.files?.length) {
        const agentDir = path.join(Instance.directory, ".agent-files")
        for (const file of result.object.files) {
          const filePath = path.join(agentDir, file.path)
          await fs.mkdir(path.dirname(filePath), { recursive: true })
          await Bun.file(filePath).write(file.content)
          log.info("wrote agent file", { path: filePath })
        }
      }

      // For non-user triggers, inject continuation as synthetic user message
      if (input.trigger !== "user") {
        const continueMsg = await Session.updateMessage({
          id: Identifier.ascending("message"),
          role: "user",
          sessionID: input.sessionID,
          time: {
            created: Date.now(),
          },
          agent: input.agent,
          model: input.model,
        })
        await Session.updatePart({
          id: Identifier.ascending("part"),
          messageID: continueMsg.id,
          sessionID: input.sessionID,
          type: "text",
          synthetic: true,
          text: result.object.continue,
          time: {
            start: Date.now(),
            end: Date.now(),
          },
        })
      }

      Bus.publish(Event.Compacted, { sessionID: input.sessionID })
      return "continue"
    } catch (e) {
      log.error("compaction error", { error: e })
      const error = MessageV2.fromError(e, { providerID: model.providerID })
      msg.error = error
      msg.time.completed = Date.now()
      await Session.updateMessage(msg)
      return "stop"
    }
  }

  export const create = fn(
    z.object({
      sessionID: Identifier.schema("session"),
      agent: z.string(),
      model: z.object({
        providerID: z.string(),
        modelID: z.string(),
      }),
      trigger: MessageV2.CompactionTrigger,
    }),
    async (input) => {
      const msg = await Session.updateMessage({
        id: Identifier.ascending("message"),
        role: "user",
        model: input.model,
        sessionID: input.sessionID,
        agent: input.agent,
        time: {
          created: Date.now(),
        },
      })
      await Session.updatePart({
        id: Identifier.ascending("part"),
        messageID: msg.id,
        sessionID: msg.sessionID,
        type: "compaction",
        trigger: input.trigger,
      })
    },
  )
}
