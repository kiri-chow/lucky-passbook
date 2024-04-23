<script setup>
import { ref, onMounted, inject } from 'vue';
import { useModal } from 'vue-final-modal'
import { getUserRatings, getBooksByRatings } from '@/assets/api';
import BookListItem from '@/components/BookListItem.vue';
import BookDetailsItem from '@/components/BookDetailsItem.vue';


const user = inject('user');
const userRatings = inject('userRatings');


const theBook = ref({});
onMounted(async () => {
  user.value = JSON.parse(localStorage.getItem('user'));
  userRatings.value = await getUserRatings(user.value.id);
});


function displayBook(val) {
  theBook.value = val;
  const { open, close } = useModal({
    component: BookDetailsItem,
    attrs: {
      title: "test",
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
    <BookListItem name="Liked Books" :userRatings="userRatings" @displayBook="displayBook" />
    <!-- <BookListItem name="List 2" :userId="1"/>
    <BookListItem name="List 3" :userId="1"/> -->
  </main>
</template>