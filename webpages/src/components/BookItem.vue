<script setup>
import SvgIcon from '@jamescoyle/vue-icon';
import { computed, onMounted, watch, inject, ref } from 'vue';
import { mdiOpenInNew, mdiStar, mdiStarOutline, mdiCloseCircleOutline } from '@mdi/js';
import { getUserRatings } from '@/assets/api';


const user = inject('user');
const userRatings = inject('userRatings');
const props = defineProps({
    book: Object,
    small: {
        type: Boolean,
        default: true,
    }
});
const genres = computed(() =>
    props.book.popular_shelves ?
        Object.entries(props.book.popular_shelves).sort((x, y) => x[1] > y[1] ? -1 : 1).slice(0, 5).map(
            x => x[0]
        ) : []
);

const title = computed(() => props.book.title ? props.book.title.slice(0, 30) + '...' : '');
const description = computed(() => props.book.description.slice(0, 100) + '...');
const emit = defineEmits(['updateRating']);
const rating = computed(() => {
    let value = {'rating': 0};
    for (let r of userRatings.value) {
        if (r.bookId === props.book.id) {
            value = r;
            break
        }
    }
    return value;
});

function updateUserRatings(data){
    for (let r of userRatings.value) {
        if (r.bookId === data.bookId) {
            r.rating = data.rating;
            break
        }
    }
}

async function cancelRating() {
    if (props.small || !rating.value.id) {
        return;
    }
    const rid = rating.value.id;
    const url = `/api/ratings/${rid}`;
    const response = await fetch(url, {method: "DELETE"});
    const json = await response.json();
    if (!response.ok){
        alert(json.message);
    } else {
        userRatings.value = userRatings.value.filter(x => x.id != rid);
    }
}

async function changeRating(event) {
    if (props.small) {
        return;
    }
    const target = event.target.closest(".item-rating a");
    const texts = target.href.split('#');
    const newRating = parseInt(texts[texts.length - 1]);

    let url = '/api/ratings/';
    let method = 'POST';
    const body = {
        userId: user.value.id,
        bookId: props.book.id,
        rating: newRating,
    }
    if (rating.value.id) {
        url = url + rating.value.id;
        method = 'PUT';
    }

    try {
        const response = await fetch(
            url, {
            headers: {
                "content-type": "application/json",
            },
            method: method,
            body: JSON.stringify(body),
        });
        const json = await response.json();
        if (!response.ok) {
            throw new Error(json.message);
        } else {
            body.id = json.id;
            if (method === 'POST'){
                userRatings.value.push(body);
            } else {
                updateUserRatings(body);
            }
        }
    } catch (err) {
        console.error(err);
        alert(err.message);
    }
}
</script>
<style>
.book {
    background: var(--color-background);
    border-radius: 0.5rem;
    box-shadow: 0.1rem 0.1rem 0.5rem;
    flex-shrink: 0;
}

.book-small {
    width: 10rem;
    font-size: 0.8rem;
    overflow-x: hidden;
}

.book-small .card-img-top {
    max-height: 12rem;
    width: fit-content;
}

.book-small .card-title {
    font-size: 1rem;
}

.item-rating-star:hover {
    color: gold;
}

.item-rating-cancel:hover {
    color: darkred;
}
</style>
<template>
    <div :class="`book mx-2 my-2 p-0 pb-1 ${small ? 'book-small' : ''}`" :id="book.id">
        <div class="d-flex justify-content-center">
            <img class="card-img-top" v-if="book.image_url" :src="book.image_url" alt="The book's image" />
        </div>
        <div :class="`card-body p-${small ? '1' : '4'}`">
            <div class="d-flex justify-content-between">
                <h5 class="card-title">{{ small ? title : book.title }}</h5>
                <a v-if="!small" target="_blank" class="card-link" :href="book.url">
                    <svg-icon type="mdi" :path="mdiOpenInNew" />
                </a>
            </div>
            <section>
                <p v-if="book.publication_year" class="card-subtitle text-secondary">
                    Published in {{ book.publication_year }}
                </p>
            </section>
            <section v-if="!(small & rating.rating === 0)" :class="`item-rating justify-content-between ${small? '' : 'd-flex'}`">
                <div :class="`d-flex justify-content-${small ? 'evenly' : 'start'}`">
                    <a :href="`#${i + 1}`" v-for="i in Array(5).keys()" @click="changeRating">
                        <svg-icon type="mdi" class="item-rating-star"
                            :path="i + 1 > rating.rating ? mdiStarOutline : mdiStar" color="orange" :size="small ? 16 : 24"/>
                    </a>
                </div>
                <a v-if="!small && rating.id" @click="cancelRating" role="button">
                    <svg-icon type="mdi" class="item-rating-cancel"
                        :path="mdiCloseCircleOutline" color="red" :size="small ? 16 : 24"/>
                </a>
            </section>
            <section>
                <p v-if="book.description && !small" class="card-text mb-2">
                    {{ description }}
                </p>
            </section>
            <section v-if="!small" class="row shelves">
                <p class="text-info mb-1" v-for="name in genres">#{{ name }}</p>
            </section>
        </div>
    </div>
</template>