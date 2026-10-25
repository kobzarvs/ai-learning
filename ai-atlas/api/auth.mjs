import { authResponse } from '../editorial-db.mjs';
export default { fetch(request) { return authResponse(request); } };
