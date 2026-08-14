/* Browser-local artifacts for the ingest debugger.
 *
 * The Retriever job tracker intentionally stores status/result data, not the
 * original upload bytes. Keep a copy in IndexedDB so a Job Detail page can
 * still show the source PDF after navigating away from the upload form. This
 * is local to the browser and is not a second backend storage path.
 */

window.NemoDebugStore = (() => {
  const DB_NAME = 'nemo-retriever-dashboard-debug';
  const STORE_NAME = 'uploads';

  function openDb() {
    return new Promise((resolve, reject) => {
      if (!window.indexedDB) { reject(new Error('IndexedDB is not available')); return; }
      const request = indexedDB.open(DB_NAME, 1);
      request.onupgradeneeded = () => {
        if (!request.result.objectStoreNames.contains(STORE_NAME)) {
          request.result.createObjectStore(STORE_NAME, { keyPath: 'jobId' });
        }
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error || new Error('IndexedDB open failed'));
    });
  }

  async function save(jobId, file) {
    if (!jobId || !file) return;
    const isPdf = file.type === 'application/pdf' || /\.pdf$/i.test(file.name || '');
    const storedFile = isPdf && file.type !== 'application/pdf'
      ? file.slice(0, file.size, 'application/pdf')
      : file;
    const db = await openDb();
    await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readwrite');
      tx.objectStore(STORE_NAME).put({
        jobId,
        file: storedFile,
        filename: file.name,
        contentType: isPdf ? 'application/pdf' : (file.type || 'application/octet-stream'),
        savedAt: new Date().toISOString(),
      });
      tx.oncomplete = resolve;
      tx.onerror = () => reject(tx.error || new Error('IndexedDB save failed'));
    });
    db.close();
  }

  async function load(jobId) {
    const db = await openDb();
    const record = await new Promise((resolve, reject) => {
      const request = db.transaction(STORE_NAME, 'readonly').objectStore(STORE_NAME).get(jobId);
      request.onsuccess = () => resolve(request.result || null);
      request.onerror = () => reject(request.error || new Error('IndexedDB read failed'));
    });
    db.close();
    return record;
  }

  async function remove(jobId) {
    if (!jobId || !window.indexedDB) return;
    const db = await openDb();
    await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readwrite');
      tx.objectStore(STORE_NAME).delete(jobId);
      tx.oncomplete = resolve;
      tx.onerror = () => reject(tx.error || new Error('IndexedDB delete failed'));
    });
    db.close();
  }

  return { save, load, remove };
})();
