import { editResponse } from '../ai-atlas/content-edit-api.mjs';
export default { fetch(request) { return editResponse(request); } };
