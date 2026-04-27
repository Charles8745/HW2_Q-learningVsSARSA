import sys
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfbase import pdfmetrics
except ImportError:
    print("reportlab not installed. Please install it.")
    sys.exit(1)

# Register Font
pdfmetrics.registerFont(TTFont('ArialUnicode', '/Library/Fonts/Arial Unicode.ttf'))

doc = SimpleDocTemplate("HW2_Report_Content.pdf", pagesize=A4)
styles = getSampleStyleSheet()

# Create Custom Styles
styles.add(ParagraphStyle(name='ChineseHeading1', fontName='ArialUnicode', fontSize=14, spaceAfter=12, spaceBefore=12))
styles.add(ParagraphStyle(name='ChineseHeading2', fontName='ArialUnicode', fontSize=12, spaceAfter=8, spaceBefore=8))
styles.add(ParagraphStyle(name='ChineseBody', fontName='ArialUnicode', fontSize=11, spaceAfter=6, leading=16))

story = []

# Section 1
story.append(Paragraph("(一) 結果分析", styles['ChineseHeading1']))

story.append(Paragraph("1. 學習表現", styles['ChineseHeading2']))
# Insert Image 1
story.append(Image("../assets/rewards_comparison.png", width=400, height=240))
story.append(Spacer(1, 10))
story.append(Paragraph("• <b>現象解釋（累積獎勵曲線）</b>：如上圖所示，在訓練過程中，SARSA 的累積獎勵快速上升並趨於穩定，整體維持在較高且平穩的水準；而 Q-learning 的累積獎勵則一直在較低的水準劇烈波動。此現象說明 SARSA 很快地學習到了安全的避險路徑，而 Q-learning 在尋找最短路徑的過程中，由於經常在懸崖邊緣探索，導致頻繁掉入懸崖而獲得極低的獎勵與巨大波動。", styles['ChineseBody']))
story.append(Paragraph("• <b>收斂速度比較</b>：在相同回合數下，Q-learning 與 SARSA 均能收斂。以策略學習觀之，Q-learning 直接更新全局最優動作價值，通常能較快學習到理論最佳路徑。然就累積獎勵收斂速度而言，SARSA 因採取避險策略，其累積獎勵曲線在初期會更快趨於穩定，且整體獎勵較高。", styles['ChineseBody']))

story.append(Paragraph("2. 策略行為", styles['ChineseHeading2']))
# Insert Image 2 & 3
story.append(Paragraph("<b>Q-learning 最終學習路徑：</b>", styles['ChineseBody']))
story.append(Image("../assets/q_learning_policy_last.png", width=450, height=150))
story.append(Spacer(1, 5))
story.append(Paragraph("<b>SARSA 最終學習路徑：</b>", styles['ChineseBody']))
story.append(Image("../assets/sarsa_policy_last.png", width=450, height=150))
story.append(Spacer(1, 10))
story.append(Paragraph("• <b>現象解釋（路徑視覺化）</b>：如上圖所示，Q-learning（上圖）選擇了緊貼懸崖邊緣的最短危險路徑；SARSA（下圖）則選擇了遠離懸崖的安全路徑。這印證了 Q-learning 傾向冒險，為追求最少步數的高報酬而走在懸崖邊緣；SARSA 則傾向保守，在更新價值時考量了探索策略導致掉入懸崖的風險，進而選擇遠離危險區域。", styles['ChineseBody']))
story.append(Paragraph("• <b>最終學習路徑與傾向分析</b>：視覺化結果明確顯示 Q-learning 學習到的是理論上的最優解（最短路徑），但風險極高；SARSA 學習到的是次優解（較長路徑），但在存在隨機探索的環境中更為安全。", styles['ChineseBody']))

story.append(Paragraph("3. 穩定性分析", styles['ChineseHeading2']))
story.append(Paragraph("• <b>學習過程波動程度</b>：結合累積獎勵曲線可見，Q-learning 學習過程中的獎勵波動劇烈，因其最優路徑鄰近懸崖，在保持探索率的情況下極易掉入懸崖；SARSA 的波動程度較小，其路徑安全，即便隨機探索亦不易導致嚴重懲罰。", styles['ChineseBody']))
story.append(Paragraph("• <b>探索對結果的影響</b>：探索機制使得 Q-learning 在訓練期間表現較差，頻繁偏離最優路徑；SARSA 則將探索帶來的潛在風險納入價值評估，其策略能容忍探索行為，展現較高穩定性。", styles['ChineseBody']))

# Section 2
story.append(Paragraph("(二) 理論比較與討論", styles['ChineseHeading1']))
story.append(Paragraph("• Q-learning 為離策略（Off-policy）方法，其更新目標為下一狀態的全局最大 Q 值，不考慮當前實際執行的探索策略。因此，它能學習到全局最佳策略（最短路徑），但在訓練過程中因未評估探索風險，容易掉入懸崖。", styles['ChineseBody']))
story.append(Paragraph("• SARSA 為同策略（On-policy）方法，其更新基於下一步實際採取的行動。這表示 SARSA 會將探索策略所帶來的隨機性與風險納入考量。因此，SARSA 學習到的策略會避開高危險區域，傾向安全路徑。", styles['ChineseBody']))
story.append(Paragraph("一般而言，Q-learning 傾向學習理論最優策略，但訓練過程具高風險；SARSA 傾向在實際探索策略下學習較安全、穩定的行為。", styles['ChineseBody']))

# Section 3
story.append(Paragraph("(三) 結論", styles['ChineseHeading1']))
story.append(Paragraph("• <b>收斂較快的方法</b>：若以策略達到最優狀態的速度為準，Q-learning 較快；若以累積獎勵趨於穩定的速度為準，SARSA 較快。", styles['ChineseBody']))
story.append(Paragraph("• <b>較穩定的方法</b>：SARSA 在訓練過程中展現較高的穩定性與較小的獎勵波動。", styles['ChineseBody']))
story.append(Paragraph("• <b>選擇情境</b>：", styles['ChineseBody']))
story.append(Paragraph("  - <b>Q-learning</b>：適用於環境允許試錯、懲罰成本可承受，或最終部署時能完全關閉探索以執行純粹最優策略的情境。", styles['ChineseBody']))
story.append(Paragraph("  - <b>SARSA</b>：適用於探索過程中的錯誤成本極高（如自動駕駛、醫療決策），或系統必須在持續探索的情況下安全運行的情境。", styles['ChineseBody']))

doc.build(story)
print("PDF created successfully as HW2_Report_Content.pdf")
