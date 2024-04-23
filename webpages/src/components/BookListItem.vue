<template>
    <div v-show="books.length" class="book-list mt-2 mb-1">
        <div class="book-list-header">
            <h4 class="book-list-title mb-0">{{ name }}</h4>
        </div>
        <div class="book-list-content d-flex justify-content-start w-100">
            <book-item v-for="book in books" :book="book" :small="true" @click="displayBook" />
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
                        <li :class="`page-item ${page === maxPage ? 'active' : ''}` ">
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
import { ref, onMounted, watch, inject } from 'vue';
import { getBooksByRatings, getBooksBySearch } from '@/assets/api';
import { mdiChevronLeft, mdiChevronRight } from '@mdi/js';
import BookItem from './BookItem.vue';


const userRatings = inject('userRatings');
const emit = defineEmits(['displayBook']);
const props = defineProps({
    name: String,
    search: {
        type: Object,
        default: {
            scope: 'all',
            keyword: '',
        },
    },
});
const page = ref(1);
const perPage = ref(20);
const maxPage = ref(null);

// read books
const books = ref([]);

async function changePage(event) {
    const target = event.target.closest('a');
    page.value = parseInt(target.innerText);
    updateBookList(props);
}

async function updateBookList(data) {
    const name = data.name.toLowerCase();
    if (name === 'liked books') {
        books.value = await getBooksByRatings(userRatings.value);
    } else if (name === 'search result' && data.search.keyword) {
        let result = await getBooksBySearch(data.search, page.value, perPage.value);
        page.value = result.page;
        perPage.value = result.perPage;
        maxPage.value = Math.ceil(result.total / perPage.value);
        books.value = result.data;
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
.page-link.disabled {
    background-color: rgba(0,0,0,0) !important;
    border:0;
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
}
</style>