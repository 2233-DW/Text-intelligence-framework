data tfidf;
    infile "D:\SASanalysis\SAS_text\python_SAS\output_jianmo_1\tfidf_matrix_2.csv"
        dlm=',' dsd truncover firstobs=2;
    length doc_id $10;  /* 增加文档标签字段 */
    array words[*] word_1-word_296;   /* 修正列偏移 */
    input doc_id words[*];  /* 先读取文档标签 */
run;

/* 1. 文档数量验证 */
proc sql noprint;
    select count(*) into :doc_count from tfidf;
quit;
%put 当前文档数：&doc_count;

%macro check_docs;
    %if &doc_count < 2 %then %do;
        %put ERROR: 需要至少2个文档进行相似度分析;
        endsas;
    %end;
%mend;
%check_docs;

/* 2. 矩阵计算优化 */
proc iml;
    use tfidf;
    read all var _NUM_ into X;
    close tfidf;
    
    /* 检查缺失值 */
    if any(X=. ) then do;
        print "错误：矩阵包含缺失值，请检查输入数据";
        abort;
    end;
    
    /* 安全计算余弦相似度 */
    norms = sqrt(X[,##]);
    if min(norms) < 1e-12 then do;
        print "警告：存在零向量文档，相似度结果可能不准确";
        norms = norms + 1e-12;  /* 防止除以零 */
    end;
    X_norm = X / norms;
    cos_sim = X_norm * X_norm`;
    
    /* 生成带标签的输出 */
    doc_names = "doc1":"doc"+strip(char(nrow(X)));
    create cos_sim_result from cos_sim[colname=doc_names rowname=doc_names];
    append from cos_sim[rowname=doc_names];
	close cos_sim_result;  /* 显式关闭数据集 */
quit;
/* 3. 结果输出 */
proc export data=cos_sim_result
    outfile="D:\SASanalysis\SAS_text\python_SAS\out_sasjisuan_1\cos_sim_result4.csv"
    dbms=csv 
    replace;
run;
/* 4. 优化报告 */
proc print data=cos_sim_result noobs;
    title "文档相似度矩阵 (精度: 0.0001)";
    id doc_names;  /* 显示行标签 */
    var doc1 doc2; /* 明确指定列 */
    format doc1 doc2 8.4; /* 统一精度 */
run;
proc print data=tfidf(obs=2);
    var doc_id word_1-word_5;
run;
proc means data=cos_sim_result min max mean std;
    var _NUMERIC_;
run;
proc print data=cos_sim_result(obs=2);
run;
/* 示例：SAS 中指定日志文件路径的代码 */
proc printto log="D:\SASanalysis\SAS_text\python_SAS\out_sasjisuan_1\sas_execution4.log";
run;
