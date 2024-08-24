from statistics import mean
from scipy import stats
from scipy.stats import mannwhitneyu

KNN = [0.7157190635451505, 0.6872909698996655, 0.6822742474916388, 0.68561872909699, 0.705685618729097, 0.6939799331103679, 0.7040133779264214, 0.6822742474916388, 0.6672240802675585, 0.6839464882943144, 0.7006688963210702, 0.6889632107023411, 0.6839464882943144, 0.7140468227424749, 0.697324414715719, 0.6956521739130435, 0.7157190635451505, 0.6755852842809364, 0.7023411371237458, 0.6939799331103679]
DT = [0.9297658862876255, 0.9264214046822743, 0.9180602006688964, 0.9247491638795987, 0.9247491638795987, 0.9264214046822743, 0.9163879598662207, 0.9163879598662207, 0.931438127090301, 0.9498327759197325, 0.9230769230769231, 0.9230769230769231, 0.9147157190635451, 0.931438127090301, 0.9230769230769231, 0.9297658862876255, 0.9347826086956522, 0.9297658862876255, 0.9381270903010034, 0.9331103678929766]
MLP = [0.802675585284281, 0.8177257525083612, 0.8060200668896321, 0.7842809364548495, 0.8327759197324415, 0.5066889632107023, 0.8210702341137124, 0.8327759197324415, 0.8277591973244147, 0.8695652173913043, 0.8277591973244147, 0.7959866220735786, 0.8411371237458194, 0.8578595317725752, 0.8110367892976589, 0.8511705685618729, 0.8093645484949833, 0.8444816053511706, 0.8327759197324415, 0.8160535117056856]
SVM = [0.8394648829431438, 0.8478260869565217, 0.8444816053511706, 0.8344481605351171, 0.8478260869565217, 0.8277591973244147, 0.8595317725752508, 0.8444816053511706, 0.8394648829431438, 0.8645484949832776, 0.8561872909698997, 0.842809364548495, 0.8394648829431438, 0.8411371237458194, 0.8545150501672241, 0.8545150501672241, 0.8561872909698997, 0.8478260869565217, 0.8678929765886287, 0.8528428093645485]
NB = [0.7725752508361204, 0.7759197324414716, 0.7775919732441472, 0.7909698996655519, 0.7692307692307693, 0.754180602006689, 0.7876254180602007, 0.7792642140468228, 0.7675585284280937, 0.8093645484949833, 0.8110367892976589, 0.7926421404682275, 0.7859531772575251, 0.8093645484949833, 0.7809364548494984, 0.7876254180602007, 0.782608695652174, 0.7675585284280937, 0.7876254180602007, 0.7892976588628763]
RS = [0.8862876254180602, 0.8963210702341137, 0.8712374581939799, 0.862876254180602, 0.8879598662207357, 0.8913043478260869, 0.8812709030100334, 0.8862876254180602, 0.8879598662207357, 0.9063545150501672, 0.9063545150501672, 0.882943143812709, 0.8729096989966555, 0.9147157190635451, 0.882943143812709, 0.8946488294314381, 0.8896321070234113, 0.8846153846153846, 0.8996655518394648, 0.8795986622073578]
VM = [0.8444816053511706, 0.8578595317725752, 0.8444816053511706, 0.842809364548495, 0.8528428093645485, 0.8193979933110368, 0.8678929765886287, 0.8494983277591973, 0.8545150501672241, 0.8762541806020067, 0.8729096989966555, 0.8478260869565217, 0.8511705685618729, 0.8712374581939799, 0.8494983277591973, 0.8729096989966555, 0.8695652173913043, 0.8561872909698997, 0.862876254180602, 0.8511705685618729]
BC = [0.862876254180602, 0.8712374581939799, 0.8612040133779264, 0.8595317725752508, 0.8879598662207357, 0.8595317725752508, 0.8812709030100334, 0.8779264214046822, 0.8745819397993311, 0.9080267558528428, 0.882943143812709, 0.8612040133779264, 0.8595317725752508, 0.9130434782608695, 0.8645484949832776, 0.9013377926421404, 0.8812709030100334, 0.8779264214046822, 0.8846153846153846, 0.8779264214046822]

knn_percentages = [round(value * 100, 5) for value in KNN]
dt_percentages = [round(value * 100, 5) for value in DT]
mlp_percentages = [round(value * 100, 5) for value in MLP]
svm_percentages = [round(value * 100, 5) for value in SVM]
nb_percentages = [round(value * 100, 5) for value in NB]
rs_percentages = [round(value * 100, 5) for value in RS]
vm_percentages = [round(value * 100, 5) for value in VM]
bc_percentages = [round(value * 100, 5) for value in BC]

sta1, pvalue1 = stats.kruskal(knn_percentages, dt_percentages, mlp_percentages, svm_percentages, nb_percentages)

print("Estatística de Kruskal-Wallis para sistema Monolíticos: ", sta1)
print(f"\nValor de P para sistemas monolíticos: {pvalue1:.5}")

if pvalue1 <= 0.05:
    print("\nHá diferença estatisticamente significativa entre os classificadores.\n")
    print("Iniciando análise Mannwhitneyu\n")


    # KNN X DT
    msta_kd, mpvalue_kd = mannwhitneyu(knn_percentages, dt_percentages, method = "exact", alternative = "two-sided")
    print("Estatística de Mannwhitneyu KNN X DT: ", msta_kd)
    print(f"\nValor de P: {mpvalue_kd:.5}")
    if mpvalue_kd <= 0.05:
        print("\nHá diferença estatisticamente significativa entre os classificadores KNN e DT.\n")
    else:
        print("\nNão há diferença estatisticamente significativa entre os classificadores KNN e DT.")
        
    # KNN x SVM
    msta_ks, mpvalue_ks = mannwhitneyu(knn_percentages, svm_percentages, method="exact", alternative="two-sided")
    print("Estatística de Mann-Whitney U KNN x SVM: ", msta_ks)
    print(f"Valor de P: {mpvalue_ks:.5}")
    if mpvalue_ks <= 0.05:
        print("Há diferença estatisticamente significativa entre os classificadores KNN e SVM.\n")
    else:
        print("Não há diferença estatisticamente significativa entre os classificadores KNN e SVM.\n")

    # KNN x NB
    msta_kn, mpvalue_kn = mannwhitneyu(knn_percentages, nb_percentages, method="exact", alternative="two-sided")
    print("Estatística de Mann-Whitney U KNN x NB: ", msta_kn)
    print(f"Valor de P: {mpvalue_kn:.5}")
    if mpvalue_kn <= 0.05:
        print("Há diferença estatisticamente significativa entre os classificadores KNN e NB.\n")
    else:
        print("Não há diferença estatisticamente significativa entre os classificadores KNN e NB.\n")

    # DT x MLP
    msta_dm, mpvalue_dm = mannwhitneyu(dt_percentages, mlp_percentages, method="exact", alternative="two-sided")
    print("Estatística de Mann-Whitney U DT x MLP: ", msta_dm)
    print(f"Valor de P: {mpvalue_dm:.5}")
    if mpvalue_dm <= 0.05:
        print("Há diferença estatisticamente significativa entre os classificadores DT e MLP.\n")
    else:
        print("Não há diferença estatisticamente significativa entre os classificadores DT e MLP.\n")

    # DT x SVM
    msta_ds, mpvalue_ds = mannwhitneyu(dt_percentages, svm_percentages, method="exact", alternative="two-sided")
    print("Estatística de Mann-Whitney U DT x SVM: ", msta_ds)
    print(f"Valor de P: {mpvalue_ds:.5}")
    if mpvalue_ds <= 0.05:
        print("Há diferença estatisticamente significativa entre os classificadores DT e SVM.\n")
    else:
        print("Não há diferença estatisticamente significativa entre os classificadores DT e SVM.\n")

    # DT x NB
    msta_dn, mpvalue_dn = mannwhitneyu(dt_percentages, nb_percentages, method="exact", alternative="two-sided")
    print("Estatística de Mann-Whitney U DT x NB: ", msta_dn)
    print(f"Valor de P: {mpvalue_dn:.5}")
    if mpvalue_dn <= 0.05:
        print("Há diferença estatisticamente significativa entre os classificadores DT e NB.\n")
    else:
        print("Não há diferença estatisticamente significativa entre os classificadores DT e NB.\n")

    # MLP x SVM
    msta_ms, mpvalue_ms = mannwhitneyu(mlp_percentages, svm_percentages, method="exact", alternative="two-sided")
    print("Estatística de Mann-Whitney U MLP x SVM: ", msta_ms)
    print(f"Valor de P: {mpvalue_ms:.5}")
    if mpvalue_ms <= 0.05:
        print("Há diferença estatisticamente significativa entre os classificadores MLP e SVM.\n")
    else:
        print("Não há diferença estatisticamente significativa entre os classificadores MLP e SVM.\n")

    # MLP x NB
    msta_mn, mpvalue_mn = mannwhitneyu(mlp_percentages, nb_percentages, method="exact", alternative="two-sided")
    print("Estatística de Mann-Whitney U MLP x NB: ", msta_mn)
    print(f"Valor de P: {mpvalue_mn:.5}")
    if mpvalue_mn <= 0.05:
        print("Há diferença estatisticamente significativa entre os classificadores MLP e NB.\n")
    else:
        print("Não há diferença estatisticamente significativa entre os classificadores MLP e NB.\n")

    # SVM x NB
    msta_sn, mpvalue_sn = mannwhitneyu(svm_percentages, nb_percentages, method="exact", alternative="two-sided")
    print("Estatística de Mann-Whitney U SVM x NB: ", msta_sn)
    print(f"Valor de P: {mpvalue_sn:.5}")
    if mpvalue_sn <= 0.05:
        print("Há diferença estatisticamente significativa entre os classificadores SVM e NB.\n")
    else:
        print("Não há diferença estatisticamente significativa entre os classificadores SVM e NB.\n")
else:
    print("\nNão há diferença estatisticamente significativa entre os classificadores Monolíticos.\n")


# realizando os mesmos testes para os multi classificadores

sta2, pvalue2 = stats.kruskal(rs_percentages, vm_percentages, bc_percentages)

print("Estatística de Kruskal-Wallis para sistema Multi-Classificadores: ", sta2)
print("\nValor de P para sistemas Multi-Classificadores: {pvalue2:.5}")

if pvalue2 <= 0.05:
    # Regra da Soma x Voto Majoritário
    msta_rv, mpvalue_rv = mannwhitneyu(rs_percentages, vm_percentages, method="exact", alternative="two-sided")
    print("Estatística de Mann-Whitney U Regra da Soma x Voto Majoritário: ", msta_rv)
    print(f"Valor de P: {mpvalue_rv:.5}")
    if mpvalue_rv <= 0.05:
        print("\nHá diferença estatisticamente significativa entre os classificadores Regra da Soma e Voto Majoritário.\n")
    else:
        print("\nNão há diferença estatisticamente significativa entre os classificadores Regra da Soma e Voto Majoritário.\n")

    # Regra da Soma x Borda Count
    msta_rb, mpvalue_rb = mannwhitneyu(rs_percentages, bc_percentages, method="exact", alternative="two-sided")
    print("Estatística de Mann-Whitney U Regra da Soma x Borda Count: ", msta_rb)
    print(f"Valor de P: {mpvalue_rb:.5}")
    if mpvalue_rb <= 0.05:
        print("\nHá diferença estatisticamente significativa entre os classificadores Regra da Soma e Borda Count.\n")
    else:
        print("\nNão há diferença estatisticamente significativa entre os classificadores Regra da Soma e Borda Count.\n")

    # Voto Majoritário x Borda Count
    msta_vb, mpvalue_vb = mannwhitneyu(vm_percentages, bc_percentages, method="exact", alternative="two-sided")
    print("Estatística de Mann-Whitney U Voto Majoritário x Borda Count: ", msta_vb)
    print(f"Valor de P: {mpvalue_vb:.5}")
    if mpvalue_vb <= 0.05:
        print("\nHá diferença estatisticamente significativa entre os classificadores Voto Majoritário e Borda Count.\n")
    else:
        print("\nNão há diferença estatisticamente significativa entre os classificadores Voto Majoritário e Borda Count.\n")
else:
    print("\nNão há diferença estatisticamente significativa entre os Multi-Classificadores.")

#Melhor monolítico X Melhor multi

media_knn = mean(knn_percentages)
media_dt = mean(dt_percentages)
media_mlp = mean(mlp_percentages)
media_svm = mean(svm_percentages)
media_nb = mean(nb_percentages)
media_rs = mean(rs_percentages)
media_vm = mean(vm_percentages)
media_bc = mean(bc_percentages)

medias_mono = {
    "KNN": (media_knn, knn_percentages),
    "DT": (media_dt, dt_percentages),
    "MLP": (media_mlp, mlp_percentages),
    "SVM": (media_svm, svm_percentages),
    "NB": (media_nb, nb_percentages)
}

medias_multi = {
    "RS": (media_rs, rs_percentages),
    "VM": (media_vm, vm_percentages),
    "BC": (media_bc, bc_percentages)
}

# Encontrando o classificador com a maior média em cada grupo
maior_classificador_mono, (maior_media_mono, lista_mono) = max(medias_mono.items(), key=lambda item: item[1][0])
maior_classificador_multi, (maior_media_multi, lista_multi) = max(medias_multi.items(), key=lambda item: item[1][0])

# Realizando o teste de Mann-Whitney U entre os maiores classificadores
sta_mm, pvalue_mm = mannwhitneyu(lista_mono, lista_multi, method="exact", alternative="two-sided")

# Imprimindo os resultados
print(f"Estatística de Mann-Whitney U {maior_classificador_mono} x {maior_classificador_multi}: {sta_mm}")
print(f"Valor de P: {pvalue_mm:.5}")

if pvalue_mm <= 0.05:
    print(f"\nHá diferença estatisticamente significativa entre os classificadores {maior_classificador_mono} e {maior_classificador_multi}.\n")
else:
    print(f"\nNão há diferença estatisticamente significativa entre os classificadores {maior_classificador_mono} e {maior_classificador_multi}.\n")