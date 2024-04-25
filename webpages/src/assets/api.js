export async function getUserRatings(userId) {
    const response = await fetch(`/api/ratings?userId=${userId}`);
    const json = await response.json();
    if (!response.ok) {
        alert(json.message);
    }
    return json;
}

export async function getBooksByRatings(ratings) {
    const books = [];
    for (let rating of ratings) {
        const response = await fetch(`/api/books/${rating.bookId}`);
        if (response.ok) {
            const json = await response.json();
            books.push(json);
        } else {
            console.log(`failed to get book ${rating.bookId}`);
        }
    }
    return books;
}

export async function getBooksByRecommendation(user_id, method) {
    const response = await fetch(`/api/books/recommend/${user_id}?method=${method}`);
    const json = await response.json();
    if (response.ok) {
        return json;
    } else {
        console.error(json.message);
        return [];
    }
}

export async function getBooksBySearch(search, userId, page = 1, perPage = 20) {
    const params = [
        `userId=${userId}`,
        `keyword=${search.keyword}`,
        `scope=${search.scope}`,
        `sortedBy=${search.sortedBy}`,
        `page=${page}`,
        `perPage=${perPage}`
    ];
    const url = `/api/books?${params.join('&')}`;
    const response = await fetch(url);
    const json = await response.json();
    if (!response.ok) {
        alert(json.message);
        return {};
    } else {
        return json;
    }
}
