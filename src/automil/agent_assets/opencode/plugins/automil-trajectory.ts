// automil-trajectory.ts
// Installed by `automil init --runtime opencode` to .opencode/plugins/
// Requires: opencode running on Bun (ships Bun runtime; $ is Bun shell API)
// Hook: tool.execute.after — fires after each tool execution
// Soft-fail via .nothrow() — never breaks opencode's tool execution chain
import { $ } from "bun"

export default function() {
    return {
        "tool.execute.after": async (
            input: { tool: string; args: Record<string, unknown>; sessionID: string },
            output: { title: string; output: string; metadata?: unknown }
        ) => {
            const nodeId = process.env.AUTOMIL_NODE_ID
            const runtime = process.env.AUTOMIL_RUNTIME ?? "opencode"

            // Soft-fail: if not in an autoMIL orchestrated session, do nothing
            if (!nodeId) return

            const event = {
                "gen_ai.provider.name": runtime,
                "gen_ai.event.name": "tool_call",
                "gen_ai.event.timestamp": new Date().toISOString(),
                "gen_ai.tool.name": input.tool,
                "gen_ai.tool.call.arguments": JSON.stringify(input.args ?? {}),
                "gen_ai.tool.call.result": typeof output.output === "string"
                    ? output.output.slice(0, 4096)
                    : JSON.stringify(output.output),
            }

            // Runtime plugins also target non-uv consumer repos, so the globally
            // installed console script is the documented uv-first exception.
            // automil trajectory record exits 0 for both success and soft-fail (D-94)
            await $`automil trajectory record ${JSON.stringify(event)}`.quiet().nothrow()
        }
    }
}
