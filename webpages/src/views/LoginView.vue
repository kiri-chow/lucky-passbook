<template>
    <div class="w-100 car d-flex justify-content-center align-items-center">
        <form @submit.prevent="login" class="col-10 col-md-6 p-3">
            <div class="mb-2">
                <label for="username" class="form-label">Name<span class="text-danger">*</span></label>
                <input class="form-control" id="username" placeholder="Please input your name" v-model="username" required />
            </div>
            <div class="d-flex justify-content-around">
                <button class="btn btn-primary" type="submit">Login</button>
            </div>
            
        </form>
    </div>
</template>
<style>
form {
    background: var(--vt-c-white);
    border-radius: 0.5rem;
}
</style>
<script setup>
import { ref, inject } from 'vue';
import { useRouter } from 'vue-router';
import { getUserRatings } from '@/assets/api';


const user = inject('user');
const userRatings = inject('userRatings');
const username = ref('');
const router = useRouter();

async function login() {
    const response = await fetch(
        '/api/login', {
        method: "POST",
        body: JSON.stringify({ "username": username.value }),
        headers: {
            'Content-Type': 'application/json',
        }
    }
    );
    const json = await response.json();
    if (!response.ok) {
        alert(json.message);
    } else {
        localStorage.setItem('user', JSON.stringify(json));
        user.value = json;
        userRatings.value = await getUserRatings(json.id);
        console.log(userRatings.value);
        router.push('/');
    }
}


</script>