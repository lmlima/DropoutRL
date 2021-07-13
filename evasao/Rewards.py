class SparsePReward:
    """
        SparsePReward

        Recompensa esparsa com retorno apenas positivo, depende apenas do estado atual.

        Retorna 1.0 no último estado do episódio, caso o aluno tenha concluido o curso;
        Retorna 0.0 para todos os demais estados.
    """

    @staticmethod
    def reward(dados, seq_id, seq_number):
        seq = dados.loc[[seq_id]]
        max_seq_number = seq.index.get_level_values(1).max()

        if (seq_number == max_seq_number) and (seq.loc[(seq_id, seq_number)]["FORMA_EVASAO_last"] == "Conclusão"):
            return 1.0
        return 0.0


class SparseNPReward:
    """
        SparseNPReward

        Recompensa esparsa com retorno positivo e negativo, depende apenas do estado atual.

        Retorna 1.0 no último estado do episódio, caso o aluno tenha CONCLUIDO o curso;
        Retorna -1.0 no último estado do episódio, caso o aluno tenha DESISTIDO o curso;
        Retorna 0.0 para todos os demais estados (estados não terminais).
    """

    @staticmethod
    def reward(dados, seq_id, seq_number):
        seq = dados.loc[[seq_id]]
        max_seq_number = seq.index.get_level_values(1).max()

        if seq_number == max_seq_number:
            if seq.loc[(seq_id, seq_number)]["FORMA_EVASAO_last"] == "Conclusão":
                return 1.0
            elif seq.loc[(seq_id, seq_number)]["FORMA_EVASAO_last"] == "Desistência":
                return -1.0
        return 0.0
