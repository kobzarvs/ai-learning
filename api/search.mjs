import { searchResponse } from '../ai-atlas/search-api.mjs';
export default { fetch(request) { return searchResponse(request); } };
