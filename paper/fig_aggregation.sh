# Searches latex doc for *.pdf lines, copies the relevant file into the local
# directory, and updates the line

count=1
cp $1 paper_for_arxiv.tex
grep -oh "\.\..*\.pdf" $1 | while read -r line ; do
    echo "Processing $line, $count"
    figname="fig_$count.pdf"
    sed -i.bak "s^$line^$figname^" paper_for_arxiv.tex
    cp $line $figname
    let count+=1
done
