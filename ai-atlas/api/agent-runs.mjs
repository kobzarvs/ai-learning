import { agentRunsResponse } from '../agent-api.mjs';
export default { fetch(request) { return agentRunsResponse(request); } };
