import { contentResponse } from '../ai-atlas/content-db.mjs';
export default { fetch(request) { return contentResponse(request); } };
