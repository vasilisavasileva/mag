import datetime
import json
import gzip
import os
import random

import torch
import streamlit as st
import datasets
import numpy as np

from dataclasses import dataclass

from knn import FaissKNeighbors
from model import RecModel

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

st.set_page_config(
    page_title="Авиавакансии",
    page_icon="✈️"
)

st.title("✈️ Вакансии для маёвцев")


@st.cache_resource
def get_vac_features():
    dataset = datasets.load_from_disk("preview/resources/avia_vac_dataset")
    dataset.set_format("numpy")
    return dataset


@st.cache_resource
def get_vac_dict():
    with open("preview/resources/avia_vac2idx.json.gz", "rb") as f:
        json_bytes = gzip.decompress(f.read())

    vac2idx = json.loads(json_bytes.decode())
    return vac2idx


@st.cache_resource
def get_vac_data():
    data = get_vac_features().to_pandas().drop(columns=["embedding"])
    return data


@st.cache_resource
def get_knn():
    knn = FaissKNeighbors(k=5)
    vac_dataset = get_vac_features()
    X = vac_dataset["embedding"]
    y = vac_dataset["vacancy_id"]
    knn.fit(X, y)
    return knn


@st.cache_resource
def get_model():
    model = RecModel(in_dim=798, out_dim=797, n_head=6)
    model.load_state_dict(torch.load("preview/resources/model.best.pth", map_location="cpu"))
    model.eval()
    model.cpu()
    return model


if not st.session_state.get("user_actions"):
    st.session_state["user_actions"] = []


@dataclass
class Action:
    vac_id: str
    delta: float
    dt: datetime.datetime
    action_weight: float


def process_click(vac_id, action):
    st.toast(f"Действие записано")
    dt = datetime.datetime.now()
    if not st.session_state["user_actions"]:
        timedelta = 0.
    else:
        timedelta = (dt - st.session_state["user_actions"][-1].dt).seconds / 60.

    st.session_state["user_actions"].append(Action(vac_id, timedelta, dt, float(action)))


def get_vac_card(row):
    with st.expander(row["name"]):
        st.header(row["name"])
        c1, c2 = st.columns(2)
        with c1:
            st.button("Отлик", key=f"{row["vacancy_id"]}_4_{random.randint(0, 1_000_000)}", on_click=process_click, args=(row["vacancy_id"], 4))
        with c2:
            st.button("В избранное", key=f"{row["vacancy_id"]}_2_{random.randint(0, 1_000_000)}", on_click=process_click, args=(row["vacancy_id"], 2))
        if row["keySkills.keySkill"] is not None:
            st.caption(", ".join(row["keySkills.keySkill"]))
        st.text(f"От: {row["compensation.from"]} {row["compensation.currencyCode"] or ""}")
        st.text(f"До: {row["compensation.to"]} {row["compensation.currencyCode"] or ""}")
        st.html(row["description"])


select_tab, rec_tab, history_tab, vac_tab = st.tabs(["Отклики", "Рекомендации", "История", "Вакансии"])


@st.cache_data
def get_first_samples(n: int):
    return get_vac_data().sample(n)


with select_tab:
    st.header("Просмотр вакансий")
    num = st.number_input(label="Количество случайных вакансий:", min_value=10, max_value=len(get_vac_data()))
    for _, row in get_first_samples(num).iterrows():
        get_vac_card(row)


def predict(X):
    with torch.inference_mode():
        pred = get_model().forward(torch.FloatTensor(X).unsqueeze(0))
    return pred.numpy()


def recommend(k_recs: int):
    session_data = []
    for action in st.session_state["user_actions"]:
        action: Action = action
        emb_idx = get_vac_dict()[action.vac_id]
        embedding = get_vac_features()[emb_idx]["embedding"] * action.action_weight
        embedding = np.hstack([np.array([action.delta]), embedding])
        session_data.append(embedding)

    if not session_data:
        return []

    batch = np.vstack(session_data)
    pred = predict(batch)
    recs = get_knn().predict(pred, k=k_recs)[0]

    rec_vacs_idxs = (get_vac_dict()[v_id] for v_id in recs)
    rec_vacs = [get_vac_features()[i] for i in rec_vacs_idxs]

    return rec_vacs


with rec_tab:
    st.header("Рекомендации вакансий")
    k = st.slider(label="Количество рекомендаций:", min_value=1, max_value=20, value=5)
    if not st.session_state["user_actions"]:
        st.info("Ваша история взаимодействий пуста. Рассмотрите вакансии на вкладке «Отклики».", icon="ℹ️")
    for v in recommend(k):
        get_vac_card(v)


def clear_history():
    st.session_state["user_actions"] = []

with history_tab:
    st.title("История откликов")
    st.button("Очистить", on_click=clear_history)
    for act in st.session_state["user_actions"]:
        st.text(f"{act.vac_id} {act.delta} {act.dt}")


with vac_tab:
    st.header("Просмотр списка вакансий")
    idx = st.number_input("Начальный индекс:", min_value=0, step=1)
    num = st.number_input("Количество:", min_value=1, step=1, max_value=100, value=30)
    st.dataframe(get_vac_data()[idx:idx+num])
