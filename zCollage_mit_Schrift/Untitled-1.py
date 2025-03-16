def main():
    # Grid-Größe: 7 Zeilen x 60 Spalten
    rows, cols = 7, 60
    grid = [["0" for _ in range(cols)] for _ in range(rows)]
    
    # Definition der Buchstaben im 5x5-Raster
    letter_patterns = {
        "S": [
            "11111",
            "10000",
            "11111",
            "00001",
            "11111"
        ],
        "T": [
            "11111",
            "00100",
            "00100",
            "00100",
            "00100"
        ],
        "A": [
            "01110",
            "10001",
            "11111",
            "10001",
            "10001"
        ],
        "R": [
            "11110",
            "10001",
            "11110",
            "10100",
            "10010"
        ],
        "D": [
            "11110",
            "10001",
            "10001",
            "10001",
            "11110"
        ],
        "U": [
            "10001",
            "10001",
            "10001",
            "10001",
            "11111"
        ]
    }
    
    # Das anzuzeigende Wort
    word = "STARDUST"  # Buchstaben: S, T, A, R, D, U, S, T
    
    # Berechnung der Gesamtbreite des Textes:
    # Für jeden Buchstaben 5 Spalten plus 1 Spalte Abstand zwischen den Buchstaben (außer nach dem letzten)
    text_width = len(word) * 5 + (len(word) - 1)
    
    # Zentriere horizontal und vertikal (Buchstaben sind 5 Zeilen hoch)
    start_col = (cols - text_width) // 2
    start_row = (rows - 5) // 2
    
    current_col = start_col
    for letter in word:
        pattern = letter_patterns.get(letter)
        if pattern is None:
            print(f"Kein Muster für den Buchstaben: {letter}")
            continue
        # Platziere den Buchstaben im Grid (5 Zeilen x 5 Spalten)
        for i in range(5):
            for j in range(5):
                if pattern[i][j] == "1":
                    grid[start_row + i][current_col + j] = "1"
        current_col += 5 + 1  # 5 Spalten für den Buchstaben plus 1 Spalte Abstand

    # Ausgabe des gesamten Grids
    for row in grid:
        print("".join(row))


if __name__ == '__main__':
    main()
