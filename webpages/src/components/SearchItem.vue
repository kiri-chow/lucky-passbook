<script setup>
import { ref, watch } from 'vue';
import BookListItem from './BookListItem.vue';


const emit = defineEmits(['displayBook']);
const search = ref({
    scope: 'all',
    keyword: '',
});
const keyword = ref('');

function searchBooks() {
    search.value.keyword = keyword.value;
}

function displayBook(val) {
    emit('displayBook', val);
}

watch(search, (newVal, oldVal) => {
    alert(newVal);
});

</script>
<style>
.search-box .form-select {
    width: 10rem;
}
</style>
<template>
<div class="search-box">
    <div class="d-flex mb-3 w-100">
        <select class="form-select flex-grow-0 flex-shrink-1" v-model="search.scope" @change="searchBooks">
            <option value="all">All</option>
            <option value="title">Title</option>
            <option value="description">Description</option>
        </select>
        <input v-model="keyword" class="form-control flex-grow-1" @change="searchBooks" placeholder="Search for books...">
    </div>
    <BookListItem name="Search Result" :search="search" @displayBook="displayBook" />
</div>
</template>