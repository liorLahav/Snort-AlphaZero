class EloSystem:
    def __init__(self, k_factor=32):
        self.k_factor = k_factor

    def get_expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update_ratings(self, rating_a, rating_b, score_a):
        expected_a = self.get_expected_score(rating_a, rating_b)

        new_rating_a = rating_a + self.k_factor * (score_a - expected_a)
        new_rating_b = rating_b + self.k_factor * ((1 - score_a) - (1 - expected_a))

        return round(new_rating_a), round(new_rating_b)