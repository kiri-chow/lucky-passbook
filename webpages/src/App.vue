<script setup>
import { inject, onMounted } from 'vue';
import { RouterLink, RouterView } from 'vue-router';
import { ModalsContainer } from 'vue-final-modal';
import { getUserRatings } from '@/assets/api';
import BannerItem from '@/components/BannerItem.vue';

const user = inject('user');
const userRatings = inject('userRatings');

onMounted(async () => {
  user.value = await JSON.parse(localStorage.getItem('user'));
  userRatings.value = await getUserRatings(user.value.id);
});

</script>
<template>
  <header class="position-sticky top-0">
    <BannerItem />
  </header>
  <div class="p-4">
    <ModalsContainer />
    <RouterView />
  </div>
</template>