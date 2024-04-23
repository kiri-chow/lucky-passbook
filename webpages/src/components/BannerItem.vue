<script setup>
import { inject } from 'vue';
import { RouterLink, useRoute, useRouter } from 'vue-router';

const route = useRoute();
const router = useRouter();
const user = inject('user');


function logout(){
    localStorage.removeItem('user');
    user.value = {};
    router.push('/login');
}

</script>
<template>
    <div class="navbar w-100 align-items-center d-flex justify-content-between">
        <div class="container-fluid">
            <h1 class="banner-title mx-2 my-0">{{ route.name }}</h1>
            <div v-if="user" class="my-0 mx-2 d-flex justify-content-end">
                <p>Hello</p>
                <div class="dropdown mx-2">
                    <a class="text-bold dropdown-toggle" aria-haspopup="true" aria-expanded="false" id="dropdown-toggle" data-bs-toggle="dropdown" role="button">
                        {{ user.name }}
                    </a>
                    <ul class="dropdown-menu dropdown-menu-end" aria-labelledby="dropdown-toggle">
                        <li><RouterLink class="dropdown-item" to="/">Home</RouterLink></li>
                        <li><RouterLink class="dropdown-item" to="/profile">Profile</RouterLink></li>
                        <li><a class="dropdown-item" @click="logout">Logout</a></li>
                    </ul>
                </div>
                <p>!</p>
            </div>
        </div>
    </div>
</template>
<style>
.dropdown-item {
    color: var(--dark) !important;
}

.navbar {
    color: white;
    background: var(--gray-dark);
}

.navbar p {
    margin-top: 0;
    margin-bottom: 0;
}

.text-bold {
    font-weight: bold;
    color: white;
}
</style>