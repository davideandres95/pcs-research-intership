# set that we use bibex and it it safe to delete .bbl files
$bibtex_use = 2;

# add glossary dependencies
add_cus_dep('acn', 'acr', 0, 'run_makeglossaries');

# calles makeglossaries with correct path spec
sub run_makeglossaries {
    my ($base_name, $path) = fileparse( $_[0] );
    return system "makeglossaries -d \"$path\" \"$base_name\"";
}

# tell latexmk that these files are generated for glossaries and can be cleared with -c
push @generated_exts, 'acr', 'alg', 'acn';
$clean_ext .= '%R.ist %R.xdy %R.synctex %R.synctex.gz %R.synctex(busy) %R.synctex.gz(busy)';
