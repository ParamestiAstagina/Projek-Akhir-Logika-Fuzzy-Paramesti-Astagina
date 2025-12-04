import streamlit as st
import pandas as pd
import numpy as np


# Definisi Data Dasar
kriteria_data = {
    'C1': {'nama': 'Harga Pembelian', 'bobot': 0.30, 'jenis': 'cost'},
    'C2': {'nama': 'Biaya Perawatan/Upgrade', 'bobot': 0.25, 'jenis': 'cost'},
    'C3': {'nama': 'Performa (CPU/GPU)', 'bobot': 0.20, 'jenis': 'benefit'},
    'C4': {'nama': 'Daya Tahan Baterai', 'bobot': 0.15, 'jenis': 'benefit'},
    'C5': {'nama': 'Kapasitas Penyimpanan', 'bobot': 0.10, 'jenis': 'benefit'}
}

alternatif_awal = {
    'A1': 'Asus Vivobook 14',
    'A2': 'Acer Aspire 5',
    'A3': 'Lenovo IdeaPad Gaming 3',
    'A4': 'HP Pavilion 14',
    'A5': 'MSI Modern 14'
}
matrix_awal = np.array([
    [3, 4, 2, 4, 2], # A1 (C1 C2 C3 C4 C5)
    [2, 3, 3, 2, 3], # A2
    [1, 1, 5, 1, 3], # A3
    [4, 2, 3, 3, 1], # A4
    [5, 5, 4, 5, 4]  # A5
], dtype=int)

if 'alternatif' not in st.session_state:
    st.session_state.alternatif = alternatif_awal
if 'matrix' not in st.session_state:
    st.session_state.matrix = matrix_awal

MAX_ALTERNATIF = 6

# --- FUNGSI UTILITY ---

def get_df_bobot():
    records = [
        {
            'Kode': k,
            'Nama Kriteria': v['nama'],
            'Jenis Kriteria': v['jenis'].capitalize(),
            'Bobot (W)': v['bobot']
        }
        for k, v in kriteria_data.items()
    ]
    df = pd.DataFrame(records)
    df.loc[df.index[-1], 'Bobot (W)'] = 1.00 - df['Bobot (W)'].iloc[:-1].sum()
    return df.style.format({'Bobot (W)': '{:.2f}'})

def get_next_kode(alternatif_dict):
    if not alternatif_dict:
        return 'A1'
    existing_nums = [int(k[1:]) for k in alternatif_dict.keys() if k.startswith('A')]
    last_num = max(existing_nums) if existing_nums else 0
    return f'A{last_num + 1}'

def get_df_matrix():
    matrix = st.session_state.matrix
    alternatif_list = list(st.session_state.alternatif.values())
    
    kriteria_cols = list(kriteria_data.keys()) 
    
    if matrix.size == 0:
        return pd.DataFrame(columns=['Alternatif'] + kriteria_cols)
        
    df_matrix = pd.DataFrame(
        matrix,
        index=alternatif_list,
        columns=kriteria_cols
    )
    return df_matrix.reset_index().rename(columns={'index': 'Alternatif'})

# --- FUNGSI CRUD ---

def create_alternatif(nama_alternatif, nilai_kriteria):
    new_kode = get_next_kode(st.session_state.alternatif)
    st.session_state.alternatif[new_kode] = nama_alternatif
    
    new_row = np.array(nilai_kriteria, dtype=int).reshape(1, -1)
    
    if st.session_state.matrix.size == 0:
        st.session_state.matrix = new_row
    else:
        st.session_state.matrix = np.vstack([st.session_state.matrix, new_row])
    
    st.success(f"✅ Alternatif **{nama_alternatif}** berhasil ditambahkan.")

def update_alternatif(old_kode, new_nama, new_nilai):
    st.session_state.alternatif[old_kode] = new_nama
    all_kode = list(st.session_state.alternatif.keys())
    idx_to_update = all_kode.index(old_kode)
    st.session_state.matrix[idx_to_update] = np.array(new_nilai, dtype=int)
    st.success(f"✏️ Alternatif **{old_kode}** berhasil diperbarui menjadi **{new_nama}**.")

def delete_alternatif(kode_to_delete):
    all_kode = list(st.session_state.alternatif.keys())
    try:
        idx_to_delete = all_kode.index(kode_to_delete)
    except ValueError:
        st.error("Kode alternatif tidak ditemukan.")
        return
    st.session_state.matrix = np.delete(st.session_state.matrix, idx_to_delete, axis=0)
    del st.session_state.alternatif[kode_to_delete]
    st.success(f"🗑️ Alternatif **{kode_to_delete}** berhasil dihapus.")

# --- FUNGSI PERHITUNGAN SPK (SAW & TOPSIS) ---

def hitung_saw(matrix, bobot, jenis, alternatif_list):
    if matrix.size == 0: return None
    R = np.zeros_like(matrix, dtype=float)
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        max_val = np.max(col)
        min_val = np.min(col)
        if jenis[j] == 'cost':
            R[:, j] = np.divide(min_val, col, out=np.zeros_like(col, dtype=float), where=col!=0)
        elif jenis[j] == 'benefit':
            R[:, j] = np.divide(col, max_val, out=np.zeros_like(col, dtype=float), where=max_val!=0)
    V = np.dot(R, bobot)
    df_saw = pd.DataFrame({'Alternatif': alternatif_list, 'Nilai (V)': V})
    df_saw['Ranking'] = df_saw['Nilai (V)'].rank(ascending=False, method='min').astype(int)
    return df_saw.sort_values(by='Ranking')

def hitung_topsis(matrix, bobot, jenis, alternatif_list):
    if matrix.size == 0: return None
    R = np.zeros_like(matrix, dtype=float)
    for j in range(matrix.shape[1]):
        pembagi = np.sqrt(np.sum(matrix[:, j]**2))
        if pembagi != 0: 
             R[:, j] = matrix[:, j] / pembagi
        
    Y = R * bobot
    A_plus = np.zeros(Y.shape[1])
    A_min = np.zeros(Y.shape[1])
    
    for j in range(Y.shape[1]):
        if jenis[j] == 'benefit':
            A_plus[j] = np.max(Y[:, j])
            A_min[j] = np.min(Y[:, j])
        elif jenis[j] == 'cost':
            A_plus[j] = np.min(Y[:, j])
            A_min[j] = np.max(Y[:, j])
            
    S_plus = np.sqrt(np.sum((Y - A_plus)**2, axis=1))
    S_min = np.sqrt(np.sum((Y - A_min)**2, axis=1))
    
    C_plus = np.divide(S_min, (S_plus + S_min), out=np.zeros_like(S_min), where=(S_plus + S_min) != 0)
    
    df_topsis = pd.DataFrame({'Alternatif': alternatif_list, 'C+': C_plus})
    df_topsis['Ranking'] = df_topsis['C+'].rank(ascending=False, method='min').astype(int)
    return df_topsis.sort_values(by='Ranking')

# --- APLIKASI STREAMLIT UTAMA ---

st.set_page_config(layout="wide")
st.title("💻 Perbandingan Metode SAW dan TOPSIS dalam Pemilihan Jenis Laptop Terbaik untuk Mahasiswa Teknik Informatika")
st.subheader("Manajemen Data Alternatif (CRUD) dan Hasil Perankingan")
st.info(f"Batasan Maksimum Alternatif: **{MAX_ALTERNATIF}**. Data awal 5 laptop sudah dimuat.")

st.header("Parameter Kriteria dan Bobot")
st.dataframe(get_df_bobot(), hide_index=True, use_container_width=True)
st.markdown("---")

st.header("1. Kelola Data Alternatif (CRUD)")

col_matrix, col_crud_forms = st.columns([2, 3])

with col_matrix:
    st.markdown("#### Matriks Keputusan Saat Ini (Nilai 1-5)")
    df_matrix_display = get_df_matrix()
    if not df_matrix_display.empty:
        st.dataframe(df_matrix_display, hide_index=True, use_container_width=True)
    else:
        st.info("Matriks data kosong.")

with col_crud_forms:
    
    current_options = {k: v for k, v in st.session_state.alternatif.items()}
    current_codes = list(current_options.keys())

    with st.expander("➕ Tambah Alternatif Baru", expanded=len(st.session_state.alternatif) < MAX_ALTERNATIF):
        if len(st.session_state.alternatif) < MAX_ALTERNATIF:
            with st.form("form_create"):
                st.success(f"Slot tersisa: **{MAX_ALTERNATIF - len(st.session_state.alternatif)}**")
                new_nama = st.text_input("Nama Alternatif Baru", key="new_nama_input")
                
                new_nilai = []
                for k, v in kriteria_data.items():
                    nilai = st.number_input(
                        f"Nilai {k} ({v['nama'].split(' ')[0]}) [1-5]", 
                        min_value=1, max_value=5, value=3, step=1, key=f"new_nilai_{k}", help="1=Paling Ideal/Terbaik untuk Cost; 5=Paling Ideal/Terbaik untuk Benefit"
                    )
                    new_nilai.append(nilai)
                
                submitted_create = st.form_submit_button("Simpan Laptop Baru")
                if submitted_create and new_nama:
                    create_alternatif(new_nama, new_nilai)
                    st.rerun()
        else:
            st.warning(f"Batas **{MAX_ALTERNATIF}** alternatif sudah tercapai. Hapus data untuk menambah yang baru.")

    if current_codes:
        with st.expander("✏️ Edit Nilai Kriteria"):
            with st.form("form_update"):
                selected_kode_update = st.selectbox("Pilih Alternatif yang Akan Diubah", current_codes, format_func=lambda x: f"{x}: {current_options[x]}", key="select_update")
                
                idx_update = current_codes.index(selected_kode_update)
                old_nama = st.session_state.alternatif[selected_kode_update]
                old_nilai = st.session_state.matrix[idx_update].tolist()
                
                new_nama_update = st.text_input("Nama Alternatif", value=old_nama, key="update_nama_input")
                
                new_nilai_update = []
                for i, (k, v) in enumerate(kriteria_data.items()):
                    nilai = st.number_input(
                        f"Nilai {k} ({v['nama'].split(' ')[0]}) [1-5]", 
                        min_value=1, max_value=5, value=old_nilai[i], step=1, key=f"update_nilai_{k}", help=f"Nilai saat ini: {old_nilai[i]}"
                    )
                    new_nilai_update.append(nilai)

                submitted_update = st.form_submit_button("Simpan Perubahan")
                if submitted_update and new_nama_update:
                    update_alternatif(selected_kode_update, new_nama_update, new_nilai_update)
                    st.rerun()

        with st.expander("🗑️ Hapus Alternatif"):
            with st.form("form_delete"):
                selected_kode_delete = st.selectbox("Pilih Alternatif yang Akan Dihapus", current_codes, format_func=lambda x: f"{x}: {current_options[x]}", key="select_delete_2")
                
                st.error(f"Hapus **{current_options[selected_kode_delete]}**? Ini akan membuka 1 slot.")
                
                submitted_delete = st.form_submit_button("Hapus Permanen")
                if submitted_delete:
                    delete_alternatif(selected_kode_delete)
                    st.rerun()
    else:
        st.info("Tambahkan data untuk diedit atau dihapus.")

st.markdown("---")
st.header("2. Hasil Perhitungan dan Perankingan")

bobot_array = np.array([data['bobot'] for data in kriteria_data.values()])
jenis_kriteria = [data['jenis'] for data in kriteria_data.values()]
alternatif_list = list(st.session_state.alternatif.values())
current_matrix = st.session_state.matrix

if current_matrix.size > 0:
    col_saw, col_topsis = st.columns(2)

    # --- Perhitungan SAW ---
    df_saw_rank = hitung_saw(current_matrix, bobot_array, jenis_kriteria, alternatif_list)
    with col_saw:
        st.markdown("### Metode SAW (Simple Additive Weighting)")
        st.table(df_saw_rank[['Ranking', 'Alternatif', 'Nilai (V)']].style.format({'Nilai (V)': '{:.4f}'}))
        
        # PENGUMUMAN PEMENANG SAW
        saw_winner = df_saw_rank.iloc[0]['Alternatif']
        saw_score = df_saw_rank.iloc[0]['Nilai (V)']
        st.success(f"🏆 Pilihan Terbaik (SAW): **{saw_winner}** dengan Nilai V = {saw_score:.4f}")

    # --- Perhitungan TOPSIS ---
    df_topsis_rank = hitung_topsis(current_matrix, bobot_array, jenis_kriteria, alternatif_list)
    with col_topsis:
        st.markdown("### Metode TOPSIS")
        st.table(df_topsis_rank[['Ranking', 'Alternatif', 'C+']].style.format({'C+': '{:.4f}'}))
        
        # PENGUMUMAN PEMENANG TOPSIS
        topsis_winner = df_topsis_rank.iloc[0]['Alternatif']
        topsis_score = df_topsis_rank.iloc[0]['C+']
        st.success(f"🏆 Pilihan Terbaik (TOPSIS): **{topsis_winner}** dengan Nilai C+ = {topsis_score:.4f}")

    st.markdown("---")

else:
    st.warning("Tidak ada data alternatif. Silakan masukkan data di bagian atas untuk melihat hasil perhitungan.")