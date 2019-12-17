/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
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