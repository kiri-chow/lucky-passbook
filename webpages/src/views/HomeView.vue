<script setup>
import { ref, onMounted, inject } from 'vue';
import { useModal } from 'vue-final-modal'
import BookListItem from '@/components/BookListItem.vue';
import BookDetailsItem from '@/components/BookDetailsItem.vue';
import SearchItem from '@/components/SearchItem.vue';


const user = inject('user');
const userRatings = inject('userRatings');


const theBook = ref({});


function displayBook(val) {
  theBook.value = val;
  const { open, close } = useModal({
    component: BookDetailsItem,
    attrs: {
      book: theBook.value,
      userRatings: userRatings.value,
      onConfirm() {
        close()
      },
    },
  });
  open();
}

</script>
<template>
  <main>
    <SearchItem @displayBook="displayBook"/>
    <BookListItem name="Our best guess" type="svd_ncf" :userRatings="userRatings" @displayBook="displayBook" />
    <BookListItem name="Similar reader choices" type="knn_content" :userRatings="userRatings" @displayBook="displayBook" />
    <BookListItem name="Similar with your last choice" type="last_like" :userRatings="userRatings" @displayBook="displayBook" />
  </main>
</template>