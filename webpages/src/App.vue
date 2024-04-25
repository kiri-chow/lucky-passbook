<script setup>
import { inject, onMounted } from 'vue';
import { RouterLink, RouterView, useRouter } from 'vue-router';
import { ModalsContainer } from 'vue-final-modal';
import { getUserRatings } from '@/assets/api';
import BannerItem from '@/components/BannerItem.vue';

const user = inject('user');
const userRatings = inject('userRatings');
const router = useRouter()

onMounted(async () => {
  const userStr = localStorage.getItem('user');
  if (userStr) {
    user.value = await JSON.parse(userStr);
    userRatings.value = await getUserRatings(user.value.id);
  } else {
    router.push('/login');
  }
});

</script>
<template>
  <header class="position-sticky top-0">
    <BannerItem />
  </header>
  <div class="app-content p-4">
    <ModalsContainer />
    <RouterView />
  </div>
</template>