import { notificationsResponse } from '../notifications-api.mjs';
export default { fetch(request) { return notificationsResponse(request); } };
