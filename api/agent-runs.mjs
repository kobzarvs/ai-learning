import { agentRunsResponse } from '../ai-atlas/agent-api.mjs';
export default { fetch(request) { return agentRunsResponse(request); } };
