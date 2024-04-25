<script setup>
import SvgIcon from '@jamescoyle/vue-icon';
import { computed, inject, ref } from 'vue';
import { mdiOpenInNew, mdiStar, mdiStarOutline, mdiCloseCircleOutline, mdiThumbDownOutline } from '@mdi/js';
import { getUserRatings } from '@/assets/api';


const user = inject('user');
const userRatings = inject('userRatings');
const props = defineProps({
    book: {
        type: Object,
        default: {},
    },
    small: {
        type: Boolean,
        default: true,
    }
});
const genres = computed(() =>
    props.book.popular_shelves ?
        Object.entries(props.book.popular_shelves).sort((x, y) => x[1] > y[1] ? -1 : 1).slice(0, 5).map(z => z[0]) :
        []
);
const imageUrl = computed(() => {
    if (props.small) {
        return props.book.image_url;
    } else {
        const path = props.book.image_url.split('/');
        const ind = path.length - 2;
        if (path[ind][path[ind].length - 1] === 'm') {
            path[ind] = path[ind].slice(0, path[ind].length - 1) + 'i';
        }
        return path.join('/');
    }
});
const title = computed(() => props.book.title ? (
    props.book.title.length > 33 ? props.book.title.slice(0, 30) + '...' : props.book.title) :
    '');
const description = computed(() => props.book.description.slice(0, 100) + '...');
const emit = defineEmits(['updateRating', 'dislikeItem']);
const rating = computed(() => {
    let value = { 'rating': 0 };
    for (let r of userRatings.value) {
        if (r.bookId === props.book.id) {
            value = r;
            break
        }
    }
    return value;
});
const isDislike = computed(() => {
    return rating.value.id && rating.value.rating == 0 ? true : false;
});
const rCount = computed(() => {
    let count = props.book.rCount;
    let postfix = '';
    for (let p of ['K', 'M']) {
        if (count < 1000) break;
        count = Math.round(count / 1000);
        postfix = p;
    }
    return count.toLocaleString() + postfix;
})

async function updateUserRatings() {
    userRatings.value = await getUserRatings(user.value.id);
}

async function cancelRating() {
    if (props.small || !rating.value.id) {
        return;
    }
    const rid = rating.value.id;
    const url = `/api/ratings/${rid}`;
    const response = await fetch(url, { method: "DELETE" });
    const json = await response.json();
    if (!response.ok) {
        alert(json.message);
    } else {
        updateUserRatings();
    }
}

async function dislikeItem() {
    if (props.small || isDislike.value) {
        return;
    }
    const url = `/api/ratings`;
    const response = await fetch(
        url, {
        method: "POST",
        body: JSON.stringify({
            userId: user.value.id,
            bookId: props.book.id,
            rating: 0,
        }),
        headers: {
            "content-type": "application/json",
        },
    });
    const json = await response.json();
    if (!response.ok) {
        alert(json.message);
    } else {
        updateUserRatings();
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
            updateUserRatings();
        }
    } catch (err) {
        console.error(err);
        alert(err.message);
    }
}
</script>
<style>
.item-rating a {
    margin-bottom: 0;
}

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

.book .card-img-top {
    width: auto;
    height: auto;
    max-width: 20rem;
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

.shelves {
    overflow-x: hidden;
}

.shelves p {
    font-size: xx-small;
}
</style>
<template>
    <div :class="`book mx-2 my-2 p-0 pb-1 ${small ? 'book-small' : ''}`" :id="book.id">
        <div class="d-flex justify-content-center">
            <img class="card-img-top" v-if="book.image_url" :src="imageUrl" alt="The book's image" />
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
            <section>
                <p v-if="book.rCount">
                    {{ `Overall ${Math.round(10 * book.rSum / book.rCount) / 10}/5 (${rCount})` }}
                </p>
            </section>
            <section>
                <p v-if="book.conf">Our confidence: {{ book.conf }}</p>
            </section>
            <section class="mb-2" v-if="!(small && rating.rating === 0 && !isDislike)"
                :class="`item-rating align-items-center justify-content-between flex-grow-1 ${small ? '' : 'd-flex'}`">
                <div v-if="!isDislike" :class="`d-flex justify-content-${small ? 'evenly' : 'start'}`">
                    <a class="d-flex align-items-center" :href="`#${i + 1}`" v-for="i in Array(5).keys()"
                        @click="changeRating">
                        <svg-icon type="mdi" class="item-rating-star"
                            :path="i + 1 > rating.rating ? mdiStarOutline : mdiStar" color="orange"
                            :size="small ? 16 : 24" />
                    </a>
                </div>
                <p class="text-danger mb-0" v-else>Not Interested</p>
                <div v-if="!small" class="d-flex align-items-center">
                    <button v-if="rating.id" @click="cancelRating" class="btn btn-outline-danger">
                        Cancel Rating
                    </button>
                    <button v-else @click="dislikeItem" :class="`btn btn-outline-${isDislike ? 'success' : 'danger'}`" :disabled="isDislike">Not Interested</button>
                </div>
            </section>
            <section>
                <p v-if="book.description && !small" class="card-text mb-2">
                    {{ description }}
                </p>
            </section>
            <section v-if="!small" class="shelves d-flex justify-content-between">
                <p class="text-secondary mb-1 mx-1" v-for="name in genres">#{{ name }}</p>
            </section>
        </div>
    </div>
</template>