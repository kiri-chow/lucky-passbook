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

export async function getBooksBySearch(search, page = 1, perPage = 20) {
    const params = [
        `keyword=${search.keyword}`,
        `scope=${search.scope}`,
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
