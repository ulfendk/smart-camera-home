from dataclasses import dataclass


@dataclass
class Detection:
    label: str
    score: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2, normalised 0..1

    @property
    def center(self) -> tuple[float, float]:
        return (self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2

    @property
    def bottom_center(self) -> tuple[float, float]:
        return (self.box[0] + self.box[2]) / 2, self.box[3]
