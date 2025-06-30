data tfidf;
    infile "D:\SASanalysis\SAS_text\python_SAS\output_jianmo_1\tfidf_matrix_2.csv"
        dlm=',' dsd truncover firstobs=2;
    length doc_id $10;  /* �����ĵ���ǩ�ֶ� */
    array words[*] word_1-word_296;   /* ������ƫ�� */
    input doc_id words[*];  /* �ȶ�ȡ�ĵ���ǩ */
run;

/* 1. �ĵ�������֤ */
proc sql noprint;
    select count(*) into :doc_count from tfidf;
quit;
%put ��ǰ�ĵ�����&doc_count;

%macro check_docs;
    %if &doc_count < 2 %then %do;
        %put ERROR: ��Ҫ����2���ĵ��������ƶȷ���;
        endsas;
    %end;
%mend;
%check_docs;

/* 2. ��������Ż� */
proc iml;
    use tfidf;
    read all var _NUM_ into X;
    close tfidf;
    
    /* ���ȱʧֵ */
    if any(X=. ) then do;
        print "���󣺾������ȱʧֵ��������������";
        abort;
    end;
    
    /* ��ȫ�����������ƶ� */
    norms = sqrt(X[,##]);
    if min(norms) < 1e-12 then do;
        print "���棺�����������ĵ������ƶȽ�����ܲ�׼ȷ";
        norms = norms + 1e-12;  /* ��ֹ������ */
    end;
    X_norm = X / norms;
    cos_sim = X_norm * X_norm`;
    
    /* ���ɴ���ǩ����� */
    doc_names = "doc1":"doc"+strip(char(nrow(X)));
    create cos_sim_result from cos_sim[colname=doc_names rowname=doc_names];
    append from cos_sim[rowname=doc_names];
	close cos_sim_result;  /* ��ʽ�ر����ݼ� */
quit;
/* 3. ������ */
proc export data=cos_sim_result
    outfile="D:\SASanalysis\SAS_text\python_SAS\out_sasjisuan_1\cos_sim_result4.csv"
    dbms=csv 
    replace;
run;
/* 4. �Ż����� */
proc print data=cos_sim_result noobs;
    title "�ĵ����ƶȾ��� (����: 0.0001)";
    id doc_names;  /* ��ʾ�б�ǩ */
    var doc1 doc2; /* ��ȷָ���� */
    format doc1 doc2 8.4; /* ͳһ���� */
run;
proc print data=tfidf(obs=2);
    var doc_id word_1-word_5;
run;
proc means data=cos_sim_result min max mean std;
    var _NUMERIC_;
run;
proc print data=cos_sim_result(obs=2);
run;
/* ʾ����SAS ��ָ����־�ļ�·���Ĵ��� */
proc printto log="D:\SASanalysis\SAS_text\python_SAS\out_sasjisuan_1\sas_execution4.log";
run;
