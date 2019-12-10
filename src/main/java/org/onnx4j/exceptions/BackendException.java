package org.onnx4j.exceptions;

public class BackendException extends ForwarderException {

	private static final long serialVersionUID = -9110594428291367746L;

	private static final int ERR_CODE_BASE = 30000;

	public enum BackendExceptionEnums implements ExceptionEnums {

		INPUTS_TYPE_CONSTRAINTS(ERR_CODE_BASE + 1, "The data type of input tensors is ilegal"),
		OP_NOT_SUPPORTED(ERR_CODE_BASE + 1, "The operation \"%s\" not supported in this backend"),;

		public int code;
		public String message;

		private BackendExceptionEnums(int code, String message) {
			this.code = code;
			this.message = message;
		}

		@Override
		public int getCode() {
			return code;
		}

		@Override
		public String getMessage() {
			return message;
		}

	}

	public BackendException(BackendExceptionEnums exceptionEnum) {
		this(exceptionEnum, new Object[] {});
	}

	public BackendException(BackendExceptionEnums exceptionEnum, Object... args) {
		super(String.format(exceptionEnum.getMessage(), args));
	}

}
