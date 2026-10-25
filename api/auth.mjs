import { authResponse } from '../ai-atlas/editorial-db.mjs';
export default { fetch(request) { return authResponse(request); } };
