import z from "zod"
import { Tool } from "./tool"
import { SessionCompaction } from "../session/compaction"
import { Session } from "../session"

export const CompactTool = Tool.define("compact", {
  description: `Trigger context compaction to summarize conversation and free context space. You'll get a fresh context with a summary - the user sees no interruption.

WHEN TO COMPACT (based on quality signals you can sense):
- You're going in circles or repeating failed approaches
- You're confused about what's current vs outdated
- Context feels noisy - old exploration, irrelevant tangents, superseded information
- You've completed a major phase and want a clean slate for something different

WHEN NOT TO COMPACT:
- Just because context is large - most providers cache context, making large conversations cheaper than you'd expect
- Mid-task or mid-debugging - you'll lose important details
- You're making good progress - don't interrupt flow

TRADEOFF: Compaction loses detail but gains clarity. Use the <context-status> utilization % the system provides, combined with your sense of context quality, to decide.

When compacting, ensure the summary captures all details needed to continue: error messages, file paths, key decisions, what worked/didn't work, and current task state.`,
  parameters: z.object({
    reason: z.string().describe("Why compaction would help at this point"),
  }),
  async execute(params, ctx) {
    const session = await Session.get(ctx.sessionID)
    if (!session) {
      return {
        title: "Compaction failed",
        metadata: { reason: params.reason },
        output: "Session not found",
      }
    }

    // Find the current user message to get model info
    const msgs = await Session.messages({ sessionID: ctx.sessionID })
    const lastUserMsg = msgs.findLast((m) => m.info.role === "user")
    if (!lastUserMsg || lastUserMsg.info.role !== "user") {
      return {
        title: "Compaction failed",
        metadata: { reason: params.reason },
        output: "No user message found",
      }
    }

    await SessionCompaction.create({
      sessionID: ctx.sessionID,
      agent: lastUserMsg.info.agent,
      model: lastUserMsg.info.model,
      trigger: "model",
    })

    return {
      title: "Compaction scheduled",
      metadata: { reason: params.reason },
      output: `Compaction triggered: ${params.reason}. The context will be compacted and the conversation will continue seamlessly.`,
    }
  },
})
