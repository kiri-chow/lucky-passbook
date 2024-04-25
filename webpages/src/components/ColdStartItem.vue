<template>
    <VueFinalModal class="book-details-modal p-4 row" content-class="col-10 col-lg-6 book-details max-w-xl rounded-lg space-y-2">
        <form @submit.prevent="uploadColdStart" class="p-4">
            <h4>New User</h4>
            <textarea required v-model="sent" class="form-control" rows="5"
                placeholder="Please describe the books you are looking for..."></textarea>
            <div class="d-flex justify-content-center mt-2">
                <button type="submit" class="btn btn-primary">Submit</button>
            </div>
        </form>
    </VueFinalModal>
</template>
<style>
.book-details-modal {
    display: flex;
    justify-content: center;
    align-items: center;
}

.book-details {
    display: flex;
    flex-direction: column;
}
</style>
<script setup>
import { VueFinalModal } from 'vue-final-modal'
import { ref } from 'vue';

const props = defineProps({
    userId: String,
    onConfirm: undefined,
});
const sent = ref('');

async function uploadColdStart() {
    const response = await fetch(
        `/api/cold_start/${props.userId}`,
        {
            method: 'POST',
            body: JSON.stringify({
                sent: sent.value,
            }),
            headers: {
                "content-type": "application/json",
            },
        }
    );
    const json = await response.json();
    if (!response.ok) {
        alert(json.message);
    } else {
        props.onConfirm();
    }
}
</script>