import { editResponse } from '../content-edit-api.mjs';
export default { fetch(request) { return editResponse(request); } };
