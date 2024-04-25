<template>
    <div v-show="books.length" class="book-list mt-2 mb-1">
        <div class="book-list-header d-flex align-items-center">
            <h4 class="book-list-title mb-0">{{ name }}</h4>
            <a @click="refreshBookList" role="button" v-if="!pending">
                <svg-icon class="refresh-icon" type="mdi" :path="mdiRefresh" :size="28" />
            </a>
            <div class="loader refresh-icon" v-else />
        </div>
        <div
            :class="`w-100 justify-content-${large ? 'around book-list-content-large' : 'start book-list-content d-flex'}`">
            <book-item v-for="book in books" :book="book" :small="true" @click="displayBook" class="d-inline-block" />
        </div>
        <div v-if="maxPage" class="mt-1 mb-2 d-flex justify-content-center">
            <nav aria-label="Page navigation example">
                <ul class="pagination">
                    <li :class="`page-item ${page === 1 ? 'active' : ''}`">
                        <a class="page-link" role="button" @click="changePage">
                            1
                        </a>
                    </li>
                    <li v-if="page > 4" class="page-link disabled" disabled>...</li>
                    <li v-for="ind in Array(5).keys()" :class="`page-item ${ind === 2 ? 'active' : ''}`">
                        <a class="page-link" role="button" v-if="1 < page - 2 + ind && page - 2 + ind < maxPage"
                            @click="changePage">
                            {{ page - 2 + ind }}
                        </a>
                    </li>
                    <!-- last page -->
                    <li v-if="page < maxPage - 3" class="page-link disabled" disabled>...</li>
                    <li v-if="1 !== maxPage" :class="`page-item ${page === maxPage ? 'active' : ''}`">
                        <a class="page-link" role="button" @click="changePage">
                            {{ maxPage }}
                        </a>
                    </li>
                </ul>
            </nav>
        </div>
    </div>
</template>
<script setup>
import SvgIcon from '@jamescoyle/vue-icon';
import { ref, onMounted, watch, inject } from 'vue';
import { getBooksByRatings, getBooksBySearch, getBooksByRecommendation } from '@/assets/api';
import { mdiRefresh } from '@mdi/js';
import BookItem from './BookItem.vue';


const user = inject('user');
const userRatings = inject('userRatings');
const emit = defineEmits(['displayBook']);
const props = defineProps({
    name: String,
    type: String,
    search: {
        type: Object,
        default: {
            scope: 'all',
            sortedBy: 'date',
            keyword: '',
        },
    },
    large: {
        type: Boolean,
        default: false,
    }
});
const page = ref(1);
const perPage = ref(10);
const maxPage = ref(null);

// read books
const books = ref([]);

async function changePage(event) {
    const target = event.target.closest('a');
    page.value = parseInt(target.innerText);
    updateBookList(props);
}


const pending = ref(false);
async function refreshBookList() {
    if (!pending.value) updateBookList(props);
}

async function updateBookList(data) {
    const name = data.name.toLowerCase();
    if (data.type === 'profile') {
        books.value = await getBooksByRatings(userRatings.value);
    } else if (data.type === 'search') {
        if (data.search.keyword) {
            let result = await getBooksBySearch(data.search, user.value.id, page.value, perPage.value);
            page.value = result.page;
            perPage.value = result.perPage;
            maxPage.value = Math.ceil(result.total / perPage.value);
            books.value = result.data;
        }
    } else {
        pending.value = true;
        try {
            books.value = await getBooksByRecommendation(user.value.id, data.type);
        } catch (err) {
            alert(err.message);
        } finally {
            pending.value = false;
        }

    }
}

onMounted(async () => {
    updateBookList(props);
});

watch(props, (newVal, oldVal) => {
    updateBookList(newVal);
})


function displayBook(event) {
    const targetId = event.target.closest('.book').id;
    const book = books.value.filter(x => x.id == targetId)[0];
    emit('displayBook', book);
}
</script>
<style>
.refresh-icon {
    color: var(--blue);
    margin-left: 0.5rem;
}


.loader {
    width: 20px;
    padding: 3px;
    aspect-ratio: 1;
    border-radius: 50%;
    background: #25b09b;
    --_m:
        conic-gradient(#0000 10%, #000),
        linear-gradient(#000 0 0) content-box;
    -webkit-mask: var(--_m);
    mask: var(--_m);
    -webkit-mask-composite: source-out;
    mask-composite: subtract;
    animation: l3 1s infinite linear;
}

@keyframes l3 {
    to {
        transform: rotate(1turn)
    }
}

@keyframes l1 {
    to {
        transform: rotate(.5turn)
    }
}

.page-link.disabled {
    background-color: rgba(0, 0, 0, 0) !important;
    border: 0;
}

.book-list-content-large {
    overflow-x: hidden;
    overflow-y: scroll;
}

.book-list-content {
    overflow-x: scroll;
    overflow-y: hidden;
}

/* width */
.book-list-content::-webkit-scrollbar {
    height: 8px;
}

.book-list-content::-webkit-scrollbar-corner {
    size: 0;
}

/* Track */
.book-list-content::-webkit-scrollbar-track {
    background: rgba(200, 200, 200, 0.5);
}

/* Handle */
.book-list-content::-webkit-scrollbar-thumb {
    background: #888;
    border-radius: 4px;
}

/* Handle on hover */
.book-list-content::-webkit-scrollbar-thumb:hover {
    background: #555;
}</style>