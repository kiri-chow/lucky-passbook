import './assets/main.css'

import { createApp, ref } from 'vue'
import App from './App.vue'
import { createVfm } from 'vue-final-modal'
import router from './router'
import Oruga from '@oruga-ui/oruga-next';
// import '@oruga-ui/theme-oruga/dist/oruga.min.css';
import "bootstrap/dist/css/bootstrap.min.css";
import "bootstrap";
import 'vue-final-modal/style.css';
import '@/assets/theme.css';

const app = createApp(App);
const vfm = createVfm();

app.use(router).use(vfm).use(Oruga)
    .provide('user', ref({}))
    .provide('userRatings', ref([]));
app.mount('#app');
