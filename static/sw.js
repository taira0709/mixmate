const CACHE_NAME = 'mixmate-v1.0.0';
const STATIC_CACHE = [
  '/',
  '/static/style.css',
  '/static/app.js',
  '/static/manifest.json'
];

// Install event - cache static resources
self.addEventListener('install', (event) => {
  console.log('[SW] Installing...');
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => {
        console.log('[SW] Caching static resources');
        return cache.addAll(STATIC_CACHE);
      })
      .then(() => {
        console.log('[SW] Installation complete');
        return self.skipWaiting();
      })
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
  console.log('[SW] Activating...');
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== CACHE_NAME) {
            console.log('[SW] Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => {
      console.log('[SW] Activation complete');
      return self.clients.claim();
    })
  );
});

// Fetch event - serve from cache with network fallback
self.addEventListener('fetch', (event) => {
  const { request } = event;

  // Skip non-GET requests and external URLs
  if (request.method !== 'GET' || !request.url.startsWith(self.location.origin)) {
    return;
  }

  // Handle audio processing requests differently (don't cache)
  if (request.url.includes('/upload') ||
      request.url.includes('/process') ||
      request.url.includes('/inspect')) {
    return; // Let network handle these
  }

  event.respondWith(
    caches.open(CACHE_NAME)
      .then((cache) => {
        return cache.match(request)
          .then((cachedResponse) => {
            if (cachedResponse) {
              console.log('[SW] Serving from cache:', request.url);
              return cachedResponse;
            }

            // Fetch from network and cache if successful
            return fetch(request)
              .then((networkResponse) => {
                // Only cache successful responses
                if (networkResponse.status === 200) {
                  cache.put(request, networkResponse.clone());
                }
                return networkResponse;
              })
              .catch((error) => {
                console.error('[SW] Fetch failed:', error);
                throw error;
              });
          });
      })
  );
});

// Handle background sync for offline audio processing (future enhancement)
self.addEventListener('sync', (event) => {
  if (event.tag === 'background-audio-process') {
    console.log('[SW] Background sync triggered');
    // Future: implement offline audio processing queue
  }
});

// Handle push notifications (future enhancement)
self.addEventListener('push', (event) => {
  console.log('[SW] Push received');
  // Future: implement push notifications for processing completion
});