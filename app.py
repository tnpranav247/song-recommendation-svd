"""TuneMatch: desktop song recommender using SVD and cosine similarity.

Run with: python app.py
This version uses Tkinter and does not start a web server or open localhost.
"""
from pathlib import Path
import threading
import tkinter as tk
from tkinter import ttk, messagebox

from engine import load_dataset, calculate_svd, recommend, song_label

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "spotify-tracks-dataset-detailed.csv"
if not DATASET_PATH.exists():
    DATASET_PATH = BASE_DIR / "songs.csv"

BG = "#101512"
PANEL = "#19211c"
FG = "#f0f4f0"
MUTED = "#aab6ad"
ACCENT = "#c6dfcc"

class TuneMatchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TuneMatch | SVD Song Discovery")
        self.root.geometry("1120x720")
        self.root.minsize(900, 600)
        self.root.configure(bg=BG)
        self.df = None
        self.features = []
        self.U = None
        self.singular_values = None
        self.filtered_indices = []
        self._configure_style()
        self._build_ui()
        self.status_var.set(f"Loading dataset: {DATASET_PATH.name} …")
        threading.Thread(target=self._load_data, daemon=True).start()

    def _configure_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=FG, font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Heading.TLabel", background=BG, foreground=FG, font=("Segoe UI", 22, "bold"))
        style.configure("PanelHeading.TLabel", background=PANEL, foreground=FG, font=("Segoe UI", 11, "bold"))
        style.configure("TButton", font=("Segoe UI", 10), padding=(12, 8))
        style.configure("Accent.TButton", background=ACCENT, foreground="#17221a", font=("Segoe UI", 10, "bold"))
        style.map("Accent.TButton", background=[("active", "#d8eadc")])
        style.configure("TEntry", fieldbackground="#0d1110", foreground=FG, padding=7)
        style.configure("TCombobox", fieldbackground="#0d1110", foreground=FG, padding=6)
        style.configure("Horizontal.TScale", background=PANEL)
        style.configure("Treeview", background="#0d1110", fieldbackground="#0d1110", foreground=FG,
                        rowheight=29, bordercolor="#303a33", font=("Segoe UI", 10))
        style.configure("Treeview.Heading", background="#222a25", foreground=FG,
                        font=("Segoe UI", 10, "bold"), padding=7)
        style.map("Treeview", background=[("selected", "#3c5b45")])

    def _build_ui(self):
        header = ttk.Frame(self.root, padding=(24, 20, 24, 12))
        header.pack(fill="x")
        ttk.Label(header, text="TuneMatch", style="Heading.TLabel").pack(anchor="w")
        ttk.Label(header, text="Discover tracks with similar musical characteristics using SVD and cosine similarity.",
                  style="Muted.TLabel").pack(anchor="w", pady=(5, 0))

        body = ttk.Frame(self.root, padding=(24, 8, 24, 18))
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body, style="Panel.TFrame", padding=16)
        left.pack(side="left", fill="y", padx=(0, 16))
        left.configure(width=300)
        left.pack_propagate(False)
        ttk.Label(left, text="FIND A TRACK", style="PanelHeading.TLabel").pack(anchor="w", pady=(0, 12))
        ttk.Label(left, text="Search by title, artist, or genre.", background=PANEL, foreground=MUTED,
                  wraplength=255).pack(anchor="w")
        self.search_var = tk.StringVar()
        search_row = ttk.Frame(left, style="Panel.TFrame")
        search_row.pack(fill="x", pady=(8, 6))
        self.search_entry = ttk.Entry(search_row, textvariable=self.search_var)
        self.search_entry.pack(side="left", fill="x", expand=True)
        self.search_entry.bind("<Return>", lambda _event: self._search())
        ttk.Button(search_row, text="Search", command=self._search).pack(side="left", padx=(6, 0))
        ttk.Label(left, text="Matching songs", style="Muted.TLabel").pack(anchor="w", pady=(6, 4))
        list_frame = ttk.Frame(left, style="Panel.TFrame")
        list_frame.pack(fill="both", expand=True)
        self.song_list = tk.Listbox(list_frame, bg="#0d1110", fg=FG, selectbackground="#3c5b45",
                                    selectforeground="white", borderwidth=0, highlightthickness=0,
                                    font=("Segoe UI", 9), activestyle="none")
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.song_list.yview)
        self.song_list.configure(yscrollcommand=scrollbar.set)
        self.song_list.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.song_list.bind("<<ListboxSelect>>", self._select_song)

        ttk.Separator(left).pack(fill="x", pady=14)
        ttk.Label(left, text="SVD COMPONENTS (k)", style="PanelHeading.TLabel").pack(anchor="w")
        self.k_var = tk.IntVar(value=4)
        self.k_scale = ttk.Scale(left, from_=1, to=9, orient="horizontal", command=self._update_k)
        self.k_scale.set(4)
        self.k_scale.pack(fill="x", pady=(7, 0))
        self.k_label = ttk.Label(left, text="4 components", background=PANEL, foreground=MUTED)
        self.k_label.pack(anchor="w")
        ttk.Label(left, text="RECOMMENDATIONS", style="PanelHeading.TLabel").pack(anchor="w", pady=(14, 0))
        self.n_var = tk.IntVar(value=5)
        self.n_scale = ttk.Scale(left, from_=5, to=15, orient="horizontal", command=self._update_n)
        self.n_scale.set(5)
        self.n_scale.pack(fill="x", pady=(7, 0))
        self.n_label = ttk.Label(left, text="5 tracks", background=PANEL, foreground=MUTED)
        self.n_label.pack(anchor="w")

        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)
        selected_panel = ttk.Frame(right, style="Panel.TFrame", padding=18)
        selected_panel.pack(fill="x")
        ttk.Label(selected_panel, text="SELECTED TRACK", style="PanelHeading.TLabel").pack(anchor="w")
        self.selected_title = ttk.Label(selected_panel, text="Loading tracks…", background=PANEL,
                                        foreground=FG, font=("Segoe UI", 17, "bold"), wraplength=650)
        self.selected_title.pack(anchor="w", pady=(9, 5))
        self.selected_artist = ttk.Label(selected_panel, text="Please wait while the dataset is prepared.",
                                         background=PANEL, foreground=MUTED)
        self.selected_artist.pack(anchor="w")
        self.stats_label = ttk.Label(selected_panel, text="Tracks indexed: —    |    Audio features: —    |    Components: 4",
                                     background=PANEL, foreground=MUTED)
        self.stats_label.pack(anchor="w", pady=(14, 0))

        action_row = ttk.Frame(right)
        action_row.pack(fill="x", pady=(18, 8))
        ttk.Label(action_row, text="SIMILAR TRACKS", style="PanelHeading.TLabel").pack(side="left")
        self.recommend_btn = ttk.Button(action_row, text="Find similar tracks", style="Accent.TButton",
                                        command=self._recommend, state="disabled")
        self.recommend_btn.pack(side="right")
        self.tree = ttk.Treeview(right, columns=("track", "artist", "genre", "similarity"), show="headings")
        for col, title, width in [("track", "Track", 230), ("artist", "Artist", 180),
                                  ("genre", "Genre", 130), ("similarity", "Cosine similarity", 125)]:
            self.tree.heading(col, text=title)
            self.tree.column(col, width=width, anchor="w" if col != "similarity" else "center")
        self.tree.pack(fill="both", expand=True)
        self.status_var = tk.StringVar(value="Starting TuneMatch…")
        ttk.Label(self.root, textvariable=self.status_var, style="Muted.TLabel", padding=(24, 0, 24, 12)).pack(anchor="w")

    def _load_data(self):
        try:
            df, features = load_dataset(DATASET_PATH)
            U, singular_values = calculate_svd(df, features)
            self.root.after(0, lambda: self._data_ready(df, features, U, singular_values))
        except Exception as exc:
            self.root.after(0, lambda: messagebox.showerror("Could not load dataset", str(exc)))
            self.root.after(0, lambda: self.status_var.set("Dataset loading failed. Check the CSV file."))

    def _data_ready(self, df, features, U, singular_values):
        self.df, self.features, self.U, self.singular_values = df, features, U, singular_values
        max_k = min(len(singular_values), 9)
        self.k_scale.configure(to=max_k)
        self.k_scale.set(min(4, max_k))
        self.k_var.set(min(4, max_k))
        self._update_k(self.k_var.get())
        self.stats_label.configure(text=f"Tracks indexed: {len(df):,}    |    Audio features: {len(features)}    |    Components: {self.k_var.get()}")
        self.status_var.set(f"Ready — indexed {len(df):,} tracks using {len(features)} audio features.")
        self._search(initial=True)
        self.recommend_btn.configure(state="normal")
        if self.filtered_indices:
            self.song_list.selection_set(0)
            self._select_song()
            self._recommend()

    def _search(self, initial=False):
        if self.df is None:
            return
        query = self.search_var.get().strip().casefold()
        if not query and initial:
            self.filtered_indices = list(range(min(300, len(self.df))))
        elif not query:
            self.filtered_indices = list(range(min(300, len(self.df))))
        else:
            title = self.df["track_name"].astype(str).str.casefold()
            artist = self.df["artists"].astype(str).str.casefold()
            genre = self.df["track_genre"].astype(str).str.casefold() if "track_genre" in self.df.columns else ""
            mask = title.str.contains(query, regex=False) | artist.str.contains(query, regex=False)
            if hasattr(genre, "str"):
                mask = mask | genre.str.contains(query, regex=False)
            self.filtered_indices = self.df.index[mask].tolist()[:500]
        self.song_list.delete(0, tk.END)
        for idx in self.filtered_indices:
            row = self.df.iloc[idx]
            self.song_list.insert(tk.END, f"{row['track_name']} — {row['artists']}")
        if self.filtered_indices:
            self.song_list.selection_clear(0, tk.END)
            self.song_list.selection_set(0)
            self._select_song()
            self.status_var.set(f"Showing {len(self.filtered_indices):,} matching track(s). Select one and click Find similar tracks.")
        else:
            self.selected_title.configure(text="No matching tracks")
            self.selected_artist.configure(text="Try a different title, artist, or genre.")
            self.status_var.set("No results found.")

    def _select_song(self, _event=None):
        if self.df is None:
            return
        selection = self.song_list.curselection()
        if not selection:
            return
        idx = self.filtered_indices[selection[0]]
        row = self.df.iloc[idx]
        self.selected_title.configure(text=str(row["track_name"]))
        self.selected_artist.configure(text=str(row["artists"]) + (f"   ·   Genre: {row['track_genre']}" if "track_genre" in self.df.columns else ""))
        self.selected_index = int(idx)

    def _update_k(self, value):
        k = max(1, int(float(value)))
        self.k_var.set(k)
        if hasattr(self, "k_label"):
            self.k_label.configure(text=f"{k} component{'s' if k != 1 else ''}")
        if self.df is not None:
            self.stats_label.configure(text=f"Tracks indexed: {len(self.df):,}    |    Audio features: {len(self.features)}    |    Components: {k}")

    def _update_n(self, value):
        n = int(round(float(value)))
        self.n_var.set(n)
        if hasattr(self, "n_label"):
            self.n_label.configure(text=f"{n} tracks")

    def _recommend(self):
        if self.df is None or not hasattr(self, "selected_index"):
            return
        try:
            result = recommend(self.df, self.U, self.singular_values,
                               self.selected_index, components=self.k_var.get(), top_n=self.n_var.get())
            for item in self.tree.get_children():
                self.tree.delete(item)
            for _, row in result.iterrows():
                self.tree.insert("", "end", values=(row["Track"], row["Artist"], row["Genre"], f"{row['Similarity']:.4f}"))
            self.status_var.set(f"Found {len(result)} recommendations using k={self.k_var.get()} SVD components.")
        except Exception as exc:
            messagebox.showerror("Recommendation error", str(exc))


def main():
    root = tk.Tk()
    TuneMatchApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
