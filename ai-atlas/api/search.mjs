import { searchResponse } from '../search-api.mjs';
export default { fetch(request) { return searchResponse(request); } };
