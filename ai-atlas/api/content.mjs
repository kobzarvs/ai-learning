import { contentResponse } from '../content-db.mjs';
export default { fetch(request) { return contentResponse(request); } };
