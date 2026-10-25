import { notesResponse } from '../editorial-db.mjs';
export default { fetch(request) { return notesResponse(request); } };
