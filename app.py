# -*- coding: utf-8 -*-
import gradio as gr
from transformers import pipeline
import torch
from codigos_traducao import codigos_linguagens

def pipeline_sum_tra(text, lang_org, lang_dest):

  language_org = codigos_linguagens[lang_org]
  language_dest = codigos_linguagens[lang_dest]

  model_name_sum = "facebook/bart-large-cnn"
  model_name_tra = 'facebook/nllb-200-distilled-600M'

  if language_org == 'eng_Latn':
    summarizer = pipeline('summarization', model = model_name_sum)
    summary = summarizer(text, max_length=200, min_length=100)[0]['summary_text']

  else:
    translator_eng = pipeline('translation', model = model_name_tra, src_lang=language_org, tgt_lang='eng_Latn')
    text_eng = translator_eng(text, max_length=512)[0]['translation_text']

    summarizer = pipeline('summarization', model = model_name_sum)
    summary = summarizer(text, max_length=200, min_length=100)[0]['summary_text']

  translator = pipeline('translation', model = model_name_tra, src_lang='eng_Latn', tgt_lang=language_dest)
  translation = translator(summary, max_length=512)[0]['translation_text']

  return translation

if __name__ == '__main__':
  cod_idiomas = list(codigos_linguagens.keys())
  theme = gr.themes.Citrus(
      primary_hue="yellow",
      neutral_hue="purple",
  )

  with gr.Blocks(theme = theme) as app:
    gr.Markdown('# Tradução e Resumo de Textos')
    gr.Markdown('Esta aplicação resume e traduz o texto inserido. Modelo de tradução: facebook/nllb-200-distilled-600M. Modelo de sumarização: facebook/bart-large-cnn')
    with gr.Row():
      with gr.Column():
        texto = gr.Textbox(lines=5, label='Texto de Entrada')
        origem = gr.Dropdown(cod_idiomas, value= 'English', label='Texto original')
        destino = gr.Dropdown(cod_idiomas, value= 'Portuguese', label='Texto final')
      with gr.Column():
        resumo = gr.Textbox(lines=5, label='Resumo')

    btn = gr.Button('Gerar Resumo')
    btn.click(
        fn=pipeline_sum_tra,
        inputs= [texto, origem, destino],
        outputs= resumo,
        )

  app.launch()
