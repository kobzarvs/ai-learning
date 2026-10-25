import { notificationsResponse } from '../ai-atlas/notifications-api.mjs';
export default { fetch(request) { return notificationsResponse(request); } };
