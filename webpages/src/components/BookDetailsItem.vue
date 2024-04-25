<script setup>
import { onMounted, computed, ref, watch } from 'vue';
import { VueFinalModal } from 'vue-final-modal'
import BookItem from './BookItem.vue';


const emit = defineEmits(['leave', 'dislikeItem']);
const props = defineProps({
    book: {
        type: Object,
        default: {},
    },
    userRatings: {
        type: Array,
        default: [],
    }
})

</script>
<style>
.book-details-modal {
    display: flex;
    justify-content: center;
    align-items: center;
}

.book-details {
    max-width: 30rem;
    display: flex;
    flex-direction: column;
}

.book-details .book {
    margin: 0 !important;
}
</style>
<template>
    <VueFinalModal class="book-details-modal p-4"
        content-class="book-details max-w-xl rounded-lg space-y-2">
        <BookItem :book="book" :userRatings="userRatings" :small="false" />
    </VueFinalModal>
</template>