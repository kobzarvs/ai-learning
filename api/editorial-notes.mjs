import { notesResponse } from '../ai-atlas/editorial-db.mjs';
export default { fetch(request) { return notesResponse(request); } };
