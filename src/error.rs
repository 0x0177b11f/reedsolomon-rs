error_chain! {
	errors {
        RowsIsZero
        ColsIsZero
        SingularMatrix
		NameTooLong
		InvalidName
		InvalidAddress
		InvalidDescriptor
	}

	foreign_links {
		Io(::std::io::Error);
		ParseNum(::std::num::ParseIntError);
	}
}
