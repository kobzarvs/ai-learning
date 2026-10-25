import { historyResponse } from '../editorial-history-api.mjs';
export default { fetch(request) { return historyResponse(request); } };
