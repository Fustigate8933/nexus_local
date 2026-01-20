<script setup>
import { ref, onMounted, computed } from "vue";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

// Search state
const query = ref("");
const searchMode = ref("hybrid");
const searchResults = ref([]);
const isSearching = ref(false);
const searchError = ref("");

// Status state
const status = ref({
  store_path: "",
  vector_embeddings: 0,
  lexical_documents: 0,
});
const isLoadingStatus = ref(true);

// Indexing state
const indexPath = ref("");
const isIndexing = ref(false);
const indexProgress = ref(null);
const indexError = ref("");
const currentFile = ref("");
const progressStats = ref({
  filesIndexed: 0,
  filesSkipped: 0,
  filesUnchanged: 0,
  chunksProcessed: 0,
  currentFileName: "",
});

// Listen to indexing progress events
onMounted(async () => {
  await loadStatus();
  
  // Listen for indexing progress events
  await listen("index-progress", (event) => {
    const data = event.payload;
    if (data.type === "file-started") {
      currentFile.value = data.path;
      progressStats.value.currentFileName = data.path.split("/").pop() || data.path;
    } else if (data.type === "file-indexed") {
      progressStats.value.filesIndexed++;
      currentFile.value = "";
    } else if (data.type === "file-skipped") {
      progressStats.value.filesSkipped++;
    } else if (data.type === "file-unchanged") {
      progressStats.value.filesUnchanged++;
    } else if (data.type === "chunk-embedded") {
      progressStats.value.chunksProcessed++;
    } else if (data.type === "page-processed") {
      // Could show page progress here
    } else if (data.type === "done") {
      isIndexing.value = false;
      loadStatus();
    } else if (data.type === "error") {
      indexError.value = data.error;
      isIndexing.value = false;
    }
  });
});

const progressPercentage = computed(() => {
  const total = progressStats.value.filesIndexed + 
                progressStats.value.filesSkipped + 
                progressStats.value.filesUnchanged;
  if (total === 0) return 0;
  return Math.round((progressStats.value.filesIndexed / total) * 100);
});

async function loadStatus() {
  isLoadingStatus.value = true;
  try {
    status.value = await invoke("get_status");
  } catch (error) {
    console.error("Failed to load status:", error);
  } finally {
    isLoadingStatus.value = false;
  }
}

async function search() {
  if (!query.value.trim()) return;
  
  isSearching.value = true;
  searchError.value = "";
  searchResults.value = [];

  try {
    const results = await invoke("search", {
      query: query.value,
      mode: searchMode.value,
      limit: 10,
    });
    searchResults.value = results;
  } catch (error) {
    searchError.value = error.toString();
  } finally {
    isSearching.value = false;
  }
}

async function indexDirectory() {
  if (!indexPath.value.trim()) {
    indexError.value = "Please enter a directory path";
    return;
  }

  isIndexing.value = true;
  indexError.value = "";
  indexProgress.value = null;
  currentFile.value = "";
  progressStats.value = {
    filesIndexed: 0,
    filesSkipped: 0,
    filesUnchanged: 0,
    chunksProcessed: 0,
    currentFileName: "",
  };

  try {
    const result = await invoke("index_directory", {
      path: indexPath.value,
      gpu: false,
      max_file_mb: 50,
      max_memory_mb: null,
    });
    indexProgress.value = result;
    await loadStatus();
  } catch (error) {
    indexError.value = error.toString();
    isIndexing.value = false;
  }
}
</script>

<template>
  <div class="min-h-screen bg-linear-to-br from-slate-50 via-blue-50 to-indigo-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900">
    <!-- Header -->
    <header class="border-b border-gray-200/50 dark:border-gray-700/50 backdrop-blur-sm bg-white/80 dark:bg-gray-900/80 sticky top-0 z-50">
      <div class="max-w-7xl mx-auto px-6 py-4">
        <div class="flex items-center justify-between">
          <div>
            <h1 class="text-3xl font-bold bg-linear-to-r from-blue-600 to-indigo-600 dark:from-blue-400 dark:to-indigo-400 bg-clip-text text-transparent">
              Nexus Local
            </h1>
            <p class="text-sm text-gray-600 dark:text-gray-400 mt-1">
              Local semantic search powered by Rust
            </p>
          </div>
          <div class="flex items-center gap-4">
            <div class="text-right">
              <div class="text-xs text-gray-500 dark:text-gray-400">Vector Embeddings</div>
              <div class="text-lg font-semibold text-gray-900 dark:text-white">
                {{ status.vector_embeddings.toLocaleString() }}
              </div>
            </div>
            <div class="h-10 w-px bg-gray-300 dark:bg-gray-600"></div>
            <div class="text-right">
              <div class="text-xs text-gray-500 dark:text-gray-400">Documents</div>
              <div class="text-lg font-semibold text-gray-900 dark:text-white">
                {{ status.lexical_documents.toLocaleString() }}
              </div>
            </div>
          </div>
        </div>
      </div>
    </header>

    <main class="max-w-7xl mx-auto px-6 py-8 space-y-6">
      <!-- Search Section -->
      <div class="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-200/50 dark:border-gray-700/50 p-6">
        <div class="space-y-4">
          <div class="flex items-center gap-2 mb-4">
            <div class="w-1 h-6 bg-linear-to-b from-blue-500 to-indigo-500 rounded-full"></div>
            <h2 class="text-xl font-semibold text-gray-900 dark:text-white">Search</h2>
          </div>
          
          <div class="flex gap-3">
            <div class="flex-1 relative">
              <input
                v-model="query"
                @keyup.enter="search"
                type="text"
                placeholder="Search your indexed documents..."
                class="w-full px-4 py-3 pl-11 border border-gray-300 dark:border-gray-600 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-500 transition-all shadow-sm"
              />
              <svg class="absolute left-3.5 top-3.5 w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
								<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
							</svg>
						</div>

						<div class="relative">
							<select
								v-model="searchMode"
								class="appearance-none w-full px-4 py-3 pr-10 border border-gray-300 dark:border-gray-600 rounded-xl bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all shadow-sm"
							>
								<option value="hybrid">Hybrid</option>
								<option value="semantic">Semantic</option>
								<option value="lexical">Lexical</option>
							</select>

							<svg
								class="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400"
								fill="none"
								stroke="currentColor"
								viewBox="0 0 24 24"
							>
								<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
							</svg>
						</div>
            <button
              @click="search"
              :disabled="isSearching || !query.trim()"
              class="px-8 py-3 bg-linear-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white rounded-xl font-medium disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg hover:shadow-xl transform hover:scale-[1.02] active:scale-[0.98]"
            >
              <span v-if="!isSearching">Search</span>
              <span v-else class="flex items-center gap-2">
                <svg class="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                  <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                  <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Searching...
              </span>
            </button>
          </div>

          <div v-if="searchError" class="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl text-red-800 dark:text-red-200 flex items-center gap-2">
            <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"></path>
            </svg>
            {{ searchError }}
          </div>

          <!-- Search Results -->
          <div v-if="searchResults.length > 0" class="mt-6 space-y-3 animate-fade-in">
            <div class="flex items-center justify-between mb-4">
              <h3 class="text-lg font-semibold text-gray-900 dark:text-white">
                Results
              </h3>
              <span class="px-3 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-200 rounded-full text-sm font-medium">
                {{ searchResults.length }} found
              </span>
            </div>
            <div
              v-for="(result, index) in searchResults"
              :key="result.doc_id"
              class="group border border-gray-200 dark:border-gray-700 rounded-xl p-5 hover:shadow-lg hover:border-blue-300 dark:hover:border-blue-600 transition-all duration-200 bg-white dark:bg-gray-700/50 backdrop-blur-sm"
            >
              <div class="flex items-start justify-between mb-3">
                <div class="flex-1 min-w-0">
                  <div class="font-medium text-gray-900 dark:text-white truncate mb-1">
                    {{ result.file_path }}
                  </div>
                  <div class="flex items-center gap-3 text-sm text-gray-600 dark:text-gray-400">
                    <span>Chunk {{ result.chunk_index }}</span>
                    <span class="px-2 py-0.5 bg-linear-to-r from-blue-100 to-indigo-100 dark:from-blue-900/30 dark:to-indigo-900/30 text-blue-700 dark:text-blue-300 rounded-md text-xs font-medium">
                      {{ result.source }}
                    </span>
                    <span class="text-xs">Score: {{ result.score.toFixed(4) }}</span>
                  </div>
                </div>
              </div>
              <div v-if="result.snippet" class="mt-3 text-sm text-gray-700 dark:text-gray-300 leading-relaxed pl-4 border-l-2 border-blue-200 dark:border-blue-700">
                "{{ result.snippet.substring(0, 300) }}{{ result.snippet.length > 300 ? '...' : '' }}"
              </div>
            </div>
          </div>

          <div v-else-if="!isSearching && query.trim()" class="text-center py-12 text-gray-500 dark:text-gray-400">
            <svg class="mx-auto h-12 w-12 mb-4 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
            </svg>
            <p>No results found. Try a different search query.</p>
          </div>
        </div>
      </div>

      <!-- Indexing Section -->
      <div class="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-200/50 dark:border-gray-700/50 p-6">
        <div class="space-y-4">
          <div class="flex items-center gap-2 mb-4">
            <div class="w-1 h-6 bg-linear-to-b from-green-500 to-emerald-500 rounded-full"></div>
            <h2 class="text-xl font-semibold text-gray-900 dark:text-white">Index Directory</h2>
          </div>
          
          <div class="flex gap-3">
            <div class="flex-1 relative">
              <input
                v-model="indexPath"
                type="text"
                placeholder="~/Documents or /path/to/directory"
                class="w-full px-4 py-3 pl-11 border border-gray-300 dark:border-gray-600 rounded-xl focus:ring-2 focus:ring-green-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-500 transition-all shadow-sm"
                :disabled="isIndexing"
              />
              <svg class="absolute left-3.5 top-3.5 w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"></path>
              </svg>
            </div>
            <button
              @click="indexDirectory"
              :disabled="isIndexing || !indexPath.trim()"
              class="px-8 py-3 bg-linear-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white rounded-xl font-medium disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg hover:shadow-xl transform hover:scale-[1.02] active:scale-[0.98]"
            >
              <span v-if="!isIndexing">Index</span>
              <span v-else class="flex items-center gap-2">
                <svg class="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                  <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                  <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Indexing...
              </span>
            </button>
          </div>

          <div v-if="indexError" class="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl text-red-800 dark:text-red-200 flex items-center gap-2">
            <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"></path>
            </svg>
            {{ indexError }}
          </div>

          <!-- Progress Indicator -->
          <div v-if="isIndexing" class="mt-6 space-y-4 animate-fade-in">
            <div class="bg-linear-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-xl p-6 border border-green-200 dark:border-green-800">
              <div class="flex items-center justify-between mb-4">
                <h3 class="font-semibold text-gray-900 dark:text-white">Indexing Progress</h3>
                <span class="text-sm font-medium text-green-700 dark:text-green-300">{{ progressPercentage }}%</span>
              </div>
              
              <!-- Progress Bar 
              <div class="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3 mb-4 overflow-hidden">
                <div 
                  class="h-full bg-linear-to-r from-green-500 to-emerald-500 rounded-full transition-all duration-300 ease-out"
                  :style="{ width: progressPercentage + '%' }"
                ></div>
              </div>
							-->

              <!-- Current File -->
              <div v-if="currentFile" class="mb-4 p-3 bg-white/60 dark:bg-gray-800/60 rounded-lg">
                <div class="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-1">
                  <svg class="w-4 h-4 animate-pulse" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M10 12a2 2 0 100-4 2 2 0 000 4z"></path>
                    <path fill-rule="evenodd" d="M.458 10C1.732 5.943 5.522 3 10 3s8.268 2.943 9.542 7c-1.274 4.057-5.064 7-9.542 7S1.732 14.057.458 10zM14 10a4 4 0 11-8 0 4 4 0 018 0z" clip-rule="evenodd"></path>
                  </svg>
                  Processing
                </div>
                <div class="text-sm font-medium text-gray-900 dark:text-white truncate">
                  {{ progressStats.currentFileName }}
                </div>
              </div>

              <!-- Stats Grid -->
              <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div class="bg-white/60 dark:bg-gray-800/60 rounded-lg p-3">
                  <div class="text-xs text-gray-600 dark:text-gray-400 mb-1">Indexed</div>
                  <div class="text-2xl font-bold text-green-600 dark:text-green-400">
                    {{ progressStats.filesIndexed }}
                  </div>
                </div>
                <div class="bg-white/60 dark:bg-gray-800/60 rounded-lg p-3">
                  <div class="text-xs text-gray-600 dark:text-gray-400 mb-1">Skipped</div>
                  <div class="text-2xl font-bold text-yellow-600 dark:text-yellow-400">
                    {{ progressStats.filesSkipped }}
                  </div>
                </div>
                <div class="bg-white/60 dark:bg-gray-800/60 rounded-lg p-3">
                  <div class="text-xs text-gray-600 dark:text-gray-400 mb-1">Unchanged</div>
                  <div class="text-2xl font-bold text-blue-600 dark:text-blue-400">
                    {{ progressStats.filesUnchanged }}
                  </div>
                </div>
                <div class="bg-white/60 dark:bg-gray-800/60 rounded-lg p-3">
                  <div class="text-xs text-gray-600 dark:text-gray-400 mb-1">Chunks</div>
                  <div class="text-2xl font-bold text-purple-600 dark:text-purple-400">
                    {{ progressStats.chunksProcessed }}
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Completion Message -->
          <div v-if="indexProgress && !isIndexing" class="mt-6 p-6 bg-linear-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-xl border border-green-200 dark:border-green-800 animate-fade-in">
            <div class="flex items-center gap-3 mb-4">
              <svg class="w-6 h-6 text-green-600 dark:text-green-400" fill="currentColor" viewBox="0 0 20 20">
                <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"></path>
              </svg>
              <h3 class="font-semibold text-gray-900 dark:text-white">Indexing Complete!</h3>
            </div>
            <div class="grid grid-cols-2 md:grid-cols-5 gap-4 text-sm">
              <div>
                <div class="text-gray-600 dark:text-gray-400">Indexed</div>
                <div class="font-semibold text-gray-900 dark:text-white">{{ indexProgress.files_indexed }}</div>
              </div>
              <div>
                <div class="text-gray-600 dark:text-gray-400">Unchanged</div>
                <div class="font-semibold text-gray-900 dark:text-white">{{ indexProgress.files_unchanged }}</div>
              </div>
              <div>
                <div class="text-gray-600 dark:text-gray-400">Skipped</div>
                <div class="font-semibold text-gray-900 dark:text-white">{{ indexProgress.files_skipped }}</div>
              </div>
              <div>
                <div class="text-gray-600 dark:text-gray-400">Chunks</div>
                <div class="font-semibold text-gray-900 dark:text-white">{{ indexProgress.chunks_indexed }}</div>
              </div>
              <div>
                <div class="text-gray-600 dark:text-gray-400">Embeddings</div>
                <div class="font-semibold text-gray-900 dark:text-white">{{ indexProgress.embeddings_stored }}</div>
              </div>
            </div>
            <div v-if="indexProgress.errors.length > 0" class="mt-4 pt-4 border-t border-green-200 dark:border-green-800">
              <div class="text-sm text-red-600 dark:text-red-400">
                {{ indexProgress.errors.length }} error(s) occurred
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>

<style>
@keyframes fade-in {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.animate-fade-in {
  animation: fade-in 0.3s ease-out;
}
</style>
