import { historyResponse } from '../ai-atlas/editorial-history-api.mjs';
export default { fetch(request) { return historyResponse(request); } };
